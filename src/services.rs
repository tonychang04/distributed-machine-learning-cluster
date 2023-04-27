use futures::{prelude::*, stream::{self, StreamExt}};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::{Duration, Instant};
use enclose::enclose;
use log::{info, warn};
use path_absolutize::*;
use rand::prelude::*;
use tarpc::context::Context;
use tarpc::{client, context, server::{self, Serve, incoming::Incoming, Channel}, tokio_serde::formats::Json};
use tch::nn::{FuncT, ModuleT};
use tch::vision::{alexnet, imagenet, resnet};
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt};
use tokio::fs::{self, File};
use tokio::process::Command;
use tokio::time;
use tokio_stream::wrappers::{ReadDirStream, LinesStream};
use crate::membership::*;
use serde::{Serialize, Deserialize};

pub const LEADER_HOSTNAMES: [&str; 3] = [
    "fa22-cs425-0801.cs.illinois.edu",
    "fa22-cs425-0803.cs.illinois.edu",
    "fa22-cs425-0808.cs.illinois.edu",
];
pub const LEADER_PORT: u16 = 8851;
pub const MEMBER_PORT: u16 = 8852;

const STORAGE_DIR: &str = "storage/";
const REMOTE_STORAGE_DIR: &str = "/home/cz74/mp3/storage/";
const USER: &str = "cz74";

#[tarpc::service]
pub trait Leader {
    async fn get(filename: String, dest_id: Id, dest_path: PathBuf) -> Option<i32>;
    async fn get_versions(filename: String, count: i32, dest_id: Id, dest_path: PathBuf) -> HashSet<i32>;
    async fn put(src_id: Id, src_path: PathBuf, filename: String) -> HashSet<Id>;
    async fn delete(filename: String);
    async fn ls(filename: String) -> Vec<(Id, Vec<i32>)>;
    async fn train(filename: String, dest_path: PathBuf);
    async fn predict(label_filename: String, model_filename: String);
    async fn jobs() -> Vec<Job>;
    async fn alive() -> bool;


    // fn serve(self) -> impl Serve
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Job {
    pub model_name: String,
    pub finished_prediction_count: i32,
    pub correct_prediction_count: i32,
    pub query_durations: Vec<Duration>,
    pub assigned_member_ids: Vec<Id>,
}

impl Job {
    pub fn new(model_name: String) -> Self {
        Job {
            model_name,
            finished_prediction_count: 0,
            correct_prediction_count: 0,
            query_durations: vec![],
            assigned_member_ids: vec![],
        }
    }

    pub fn add_query_result(&mut self, correct: bool, duration: Duration) {
        self.finished_prediction_count += 1;
        if correct {
            self.correct_prediction_count += 1;
        }
        self.query_durations.push(duration);
    }
}

pub struct LeaderState {
    /// filename -> id -> versions
    pub directory: Mutex<HashMap<String, HashMap<Id, BTreeSet<i32>>>>,
    pub member: Arc<MemberState>,
    pub membership: Arc<MembershipService>,
    pub resnet18: Arc<Mutex<Job>>,
    pub alexnet: Arc<Mutex<Job>>,
    pub labels: Vec<(String, String)>,
}

pub type LeaderServer = Arc<LeaderState>;

#[tarpc::server]
impl Leader for LeaderServer {
    async fn get(self, _: Context, filename: String, dest_id: Id, dest_path: PathBuf) -> Option<i32> {
        let latest_version = self.latest_version(&filename);
        self.get_version(filename, latest_version, dest_id, dest_path).await.map(|_| latest_version)
    }

    async fn get_versions(self, _: Context, filename: String, count: i32, dest_id: Id, dest_path: PathBuf) -> HashSet<i32> {
        let latest_version = self.latest_version(&filename);
        if latest_version == 0 { return HashSet::new(); }
        let server = self.clone();
        // call get_version for last count versions concurrently
        stream::iter((1..=latest_version).rev().take(count as usize))
            .filter_map(enclose! { (filename, dest_id, dest_path) move |version| {
                let dest_path = dest_path.with_file_name(format!("v{}.{}", version, dest_path.file_name().unwrap().to_str().unwrap()));
                server.clone().get_version(filename.clone(), version, dest_id.clone(), dest_path)
                    .map(move |o| o.map(|_| version))
            }})
            .collect()
            .await
    }

    async fn put(self, _: Context, src_id: Id, src_path: PathBuf, filename: String) -> HashSet<Id> {
        let version = self.latest_version(&filename) + 1;
        self.put_version((src_id, src_path), filename, version).await
    }

    async fn delete(self, _: Context, filename: String) {
        let mut directory = self.directory.lock().unwrap();
        directory.remove(&filename);
    }

    async fn ls(self, _: Context, filename: String) -> Vec<(Id, Vec<i32>)> {
        let active_ids = self.membership.active_ids();
        self.directory.lock().unwrap()
            .get(&filename)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|(id, _)| active_ids.contains(id))
            .map(|(id, versions)| (id, versions.into_iter().collect()))
            .collect()
    }

    async fn train(self, _: Context, filename: String, dest_path: PathBuf) {
        let latest_version = self.latest_version(&filename);
        let _ = stream::iter(self.membership.active_ids()).map(|id| {
            self.clone().get_version(filename.clone(), latest_version, id.clone(), dest_path.clone())
        }).buffer_unordered(10).for_each(|_| async {}).await;
    }

    async fn predict(self, _: Context, label_filename: String, model_filename: String) {
        tokio::join!(
            self.run_job(self.resnet18.clone()),
            self.run_job(self.alexnet.clone()),
        );
    }

    async fn jobs(self, _: Context) -> Vec<Job> {
        vec![self.resnet18.lock().unwrap().clone(), self.alexnet.lock().unwrap().clone()]
    }

    async fn alive(self, _: Context) -> bool {
        true
    }
}

impl LeaderState {
    pub async fn new(membership: Arc<MembershipService>, member: Arc<MemberState>) -> Arc<Self> {
        let state = Arc::new(LeaderState {
            directory: Default::default(),
            membership,
            member,
            resnet18: Arc::new(Mutex::new(Job::new("resnet18".to_owned()))),
            alexnet: Arc::new(Mutex::new(Job::new("alexnet".to_owned()))),
            labels: {
                let label_file = File::open("synset_words.txt").await.unwrap();
                let label_file = io::BufReader::new(label_file);
                LinesStream::new(label_file.lines())
                    .filter_map(|line| async move { line.ok() })
                    .map(|line| {
                        let mut split = line.split_whitespace();
                        let filename = split.next().unwrap().to_owned();
                        // label is the rest of the line
                        let label = split.collect::<Vec<_>>().join(" ");
                        (filename, label)
                    })
                    .collect()
                    .await
            },
        });
        tokio::spawn(enclose! { (state) async move {
            // repeatedly try to make sure all files have 4 active replicas
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(3));
            interval.tick().await;
            loop {
                interval.tick().await;
                let filenames = state.directory.lock().unwrap().keys().cloned().collect::<Vec<_>>();
                for filename in filenames {
                    let version = state.latest_version(&filename);
                    state.put_version(None, filename, version).await;
                }
            }
        }});
        tokio::spawn(enclose! { (state) async move {
            // repeatedly assign members to jobs
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(3));
            interval.tick().await;
            loop {
                interval.tick().await;
                let active_ids = state.membership.active_ids().into_iter().collect::<BTreeSet<_>>();
                let mut job1 = state.resnet18.lock().unwrap();
                job1.assigned_member_ids = active_ids.iter().take(active_ids.len() / 2).cloned().collect::<Vec<_>>();
                let mut job2 = state.alexnet.lock().unwrap();
                job2.assigned_member_ids = active_ids.iter().skip(active_ids.len() / 2).cloned().collect::<Vec<_>>();
            }
        }});
        tokio::spawn(enclose! { (state) async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(3));
            let hostname = state.membership.id().hostname();
            let mut last_leader_hostname = state.member.leader_hostname();
            interval.tick().await;
            loop {
                interval.tick().await;
                let leader_hostname = state.member.leader_hostname();

                if last_leader_hostname != hostname && leader_hostname == hostname {
                    // we just became leader
                    if !state.resnet18.lock().unwrap().query_durations.is_empty() {
                        tokio::spawn(enclose! { (state) async move {
                            state.predict(context::current(), "".to_owned(), "".to_owned()).await;
                        }});
                    }
                } else if leader_hostname != hostname {
                    // copy jobs from current leader
                    let Ok(leader) = LeaderClient::spawn(&leader_hostname).await else { continue; };
                    let Ok(jobs) = leader.jobs(context::current()).await else { continue; };
                    let mut resnet18 = state.resnet18.lock().unwrap();
                    let mut alexnet = state.alexnet.lock().unwrap();
                    *resnet18 = jobs[0].clone();
                    *alexnet = jobs[1].clone();
                }

                last_leader_hostname = leader_hostname;
            }
        }});
        state
    }

    async fn scp(&self, src_hostname: &str, src_path: &Path, dest_hostname: &str, dest_path: &Path) -> ExitStatus {
        info!("scp {}:{} -> {}:{}: Starting", src_hostname, src_path.display(), dest_hostname, dest_path.display());
        let output = Command::new("scp")
            .arg(self.scp_location(src_hostname, src_path))
            .arg(self.scp_location(dest_hostname, dest_path))
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
            .await
            .expect("Failed to run or wait for scp");
        let message = format!("scp {}:{} -> {}:{}: Ended with status {}", src_hostname, src_path.display(), dest_hostname, dest_path.display(), output.status);
        if output.status.success() {
            info!("{}", message);
        } else {
            warn!("{}", message);
            warn!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
        }
        output.status
    }

    fn scp_location(&self, hostname: &str, path: &Path) -> String {
        // return an absolute local path if possible
        if hostname == self.membership.id().hostname() {
            if let Ok(path) = path.absolutize() {
                return path.display().to_string();
            }
        }
        format!("{}@{}:{}", USER, hostname, path.display())
    }

    /// Returns the latest version of the file.
    /// If the file does not exist or no versions are found, returns 0.
    fn latest_version(&self, filename: &str) -> i32 {
        let directory = self.directory.lock().unwrap();
        let default = HashMap::new();
        let map = directory.get(filename).unwrap_or(&default);
        map.values().flat_map(|vs| vs.iter()).max().cloned().unwrap_or(0)
    }

    async fn get_version(self: Arc<Self>, filename: String, version: i32, dest_id: Id, dest_path: PathBuf) -> Option<Id> {
        if version == 0 { return None; }
        let src_ids = {
            let directory = self.directory.lock().unwrap();
            let default = HashMap::new();
            let map = directory.get(&filename).unwrap_or(&default);
            map.iter()
                .filter(|(_, vs)| vs.contains(&version))
                .map(|(id, _)| id.clone())
                .collect::<Vec<_>>()
        };
        let src_path = Path::new(REMOTE_STORAGE_DIR)
            .join(MemberState::storage_filename(&filename, version));

        for src_id in src_ids {
            let scp_result = self.scp(&src_id.hostname(), &src_path, &dest_id.hostname(), &dest_path).await;
            if scp_result.success() {
                return Some(src_id);
            }
        }

        None
    }

    /// Try to ensure 4 active replicas of the file with an optional source location.
    /// If a source location is not provided, it will be chosen among the current active replicas.
    /// Returns locations of the replicas, including locations of the replicas that already existed.
    async fn put_version(&self, src: impl Into<Option<(Id, PathBuf)>>, filename: String, version: i32) -> HashSet<Id> {
        if version == 0 { return HashSet::new(); }

        let storage_filename = MemberState::storage_filename(&filename, version);

        let active_ids: HashSet<Id> = self.membership.active_ids().into_iter().collect();
        if active_ids.len() == 0 { return HashSet::new(); }

        let current_replica_ids = {
            let directory = self.directory.lock().unwrap();
            let default = HashMap::new();
            let map = directory.get(&filename).unwrap_or(&default);
            map.iter()
                .filter(|(id, vs)| active_ids.contains(id) && vs.contains(&version))
                .map(|(id, _)| id.clone())
                .collect::<HashSet<_>>()
        };

        if current_replica_ids.len() >= 4 {
            return current_replica_ids;
        }

        let (src_id, src_path) = match src.into() {
            Some(src) => src,
            None if !current_replica_ids.is_empty() => {
                let src_id = current_replica_ids.iter().next().unwrap().clone();
                let src_path = Path::new(REMOTE_STORAGE_DIR)
                    .join(MemberState::storage_filename(&filename, version));
                (src_id, src_path)
            }
            _ => {
                warn!("put_version: No source location provided and no current replicas found for file {}. We may have lost the file.", filename);
                return current_replica_ids;
            }
        };

        let new_replica_ids = {
            let mut hasher = DefaultHasher::new();
            filename.hash(&mut hasher);
            let hash = hasher.finish();

            // don't send to current replicas
            let ids_to_choose_from = active_ids.iter()
                .filter(|id| !current_replica_ids.contains(id))
                .collect::<Vec<_>>();

            if ids_to_choose_from.is_empty() { return current_replica_ids; }

            let mut result = HashSet::new();
            for i in 0..(4 - current_replica_ids.len()) {
                let index = (hash as usize + i) % ids_to_choose_from.len();
                result.insert(ids_to_choose_from[index].clone());
            }
            result
        };

        // send to new replicas concurrently and record received ids
        let received_replica_ids = stream::iter(new_replica_ids)
            .map(|dest_id| async {
                let dest_path = Path::new(REMOTE_STORAGE_DIR).join(&storage_filename);
                let scp_result = self.scp(&src_id.hostname(), &src_path, &dest_id.hostname(), &dest_path).await;
                (dest_id, scp_result)
            })
            .buffer_unordered(10)
            .filter_map(|(id, status)| async move {
                if status.success() { Some(id) } else { None }
            })
            .map(|id| async {
                let dest_hostname = id.address.split(':').next().unwrap();
                let Ok(member) = MemberClient::spawn(dest_hostname).await else {
                    warn!("put_version: Failed to connect to {}", id.address);
                    return None;
                };
                match member.receive(context::current(), filename.clone(), version).await {
                    Ok(_) => {
                        Some(id)
                    }
                    Err(e) => {
                        warn!("put_version: Failed to receive: {}", e);
                        None
                    }
                }
            })
            .buffer_unordered(10)
            .filter_map(|id| async { id })
            .collect::<HashSet<_>>()
            .await;

        let mut directory = self.directory.lock().unwrap();
        let map = directory.entry(filename).or_insert_with(Default::default);
        for id in &received_replica_ids {
            map.entry(id.clone()).or_insert_with(Default::default).insert(version);
        }

        current_replica_ids.union(&received_replica_ids).cloned().collect()
    }

    async fn run_job(&self, job: Arc<Mutex<Job>>) {
        let mut interval = time::interval(Duration::from_secs_f32(0.5));
        let model_name = job.lock().unwrap().model_name.clone();
        let finished_prediction_count = job.lock().unwrap().finished_prediction_count.clone() as usize;
        for (filename, true_label) in self.labels[finished_prediction_count..].to_owned() {
            interval.tick().await;

            // select a random assigned id
            let assigned_ids = job.lock().unwrap().assigned_member_ids.clone();
            let Some(id) = assigned_ids.into_iter().choose(&mut thread_rng()) else { continue; };

            tokio::spawn(enclose! { (job, model_name) async move {
                let start = Instant::now();
                let member = MemberClient::spawn(&id.hostname()).await?;
                let Some((probability, predicted_label)) = member.predict(context::current(), model_name.clone(), vec![filename.clone()]).await?.and_then(|v| v.into_iter().next()) else {
                    return Ok(());
                };
                job.lock().unwrap().add_query_result(predicted_label == true_label, start.elapsed());
                if predicted_label == true_label {
                    println!("{} - {}: {} ({:.2}%)", &model_name, filename, predicted_label, probability * 100.0);
                } else {
                    println!("{} - {}: {} ({:.2}%) (should be {})", &model_name, filename, predicted_label, probability * 100.0, true_label);
                }
                Ok::<_, anyhow::Error>(())
            }});
        }
    }
}

impl LeaderClient {
    pub async fn spawn(hostname: &str) -> io::Result<Self> {
        let transport = tarpc::serde_transport::tcp::connect((hostname, LEADER_PORT), Json::default).await?;
        Ok(LeaderClient::new(client::Config::default(), transport).spawn())
    }
}

#[tarpc::service]
pub trait Member {
    async fn get_latest_version(filename: String) -> Option<i32>;
    async fn receive(filename: String, version: i32);
    async fn predict(model_name: String, input_ids: Vec<String>) -> Option<Vec<(f64, String)>>;
}

pub struct MemberState {
    // filename -> versions
    pub files: Mutex<HashMap<String, BTreeSet<i32>>>,
    pub membership: Arc<MembershipService>,
    pub leader_hostname: Mutex<String>,
    pub resnet18: Mutex<Box<dyn ModuleT>>,
    pub alexnet: Mutex<Box<dyn ModuleT>>,
}

pub type MemberServer = Arc<MemberState>;

#[tarpc::server]
impl Member for MemberServer {
    async fn get_latest_version(self, _: Context, filename: String) -> Option<i32> {
        let files = self.files.lock().unwrap();
        let default = BTreeSet::new();
        let versions = files.get(&filename).unwrap_or(&default);
        versions.iter().next_back().cloned()
    }

    async fn receive(self, _: Context, filename: String, version: i32) {
        let mut files = self.files.lock().unwrap();
        files.entry(filename).or_insert_with(BTreeSet::new).insert(version);
    }

    async fn predict(self, _: Context, model_name: String, input_ids: Vec<String>) -> Option<Vec<(f64, String)>> {
        let input_folder = PathBuf::from("test_files/imagenet_1k/train");
        let model = match model_name.as_str() {
            "resnet18" => &self.resnet18,
            "alexnet" => &self.alexnet,
            _ => return None,
        };

        let mut result = Vec::new();
        for input_id in input_ids {
            let Some(image_path) = ReadDirStream::new(fs::read_dir(input_folder.join(input_id)).await.unwrap())
                .filter_map(|entry| async { entry.ok() })
                .map(|entry| entry.path())
                .boxed()
                .next()
                .await else { continue; };

            let image = imagenet::load_image_and_resize(image_path, 224, 224).ok()?;
            let output = model.lock().unwrap().forward_t(&image.unsqueeze(0), false).softmax(-1, tch::Kind::Float);
            result.push(imagenet::top(&output, 1).into_iter().next().unwrap());
        }
        Some(result)
    }
}


impl MemberState {
    pub async fn shared(membership: Arc<MembershipService>) -> io::Result<Arc<Self>> {
        // recreate storage directory
        if let Err(e) = fs::remove_dir_all(STORAGE_DIR).await {
            if e.kind() != io::ErrorKind::NotFound { return Err(e); }
        }
        fs::create_dir(STORAGE_DIR).await?;

        let state = Arc::new(Self {
            files: Default::default(),
            membership,
            leader_hostname: Mutex::new(LEADER_HOSTNAMES[0].to_owned()),
            resnet18: {
                let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
                let resnet18 = resnet::resnet18(&vs.root(), imagenet::CLASS_COUNT);
                vs.load("pretrained_models/resnet18.ot").unwrap();
                Mutex::new(Box::new(resnet18))
            },
            alexnet: {
                let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
                let alexnet = alexnet::alexnet(&vs.root(), imagenet::CLASS_COUNT);
                vs.load("pretrained_models/alexnet.ot").unwrap();
                Mutex::new(Box::new(alexnet))
            },
        });

        tokio::spawn(enclose! { (state) async move {
            // periodically check if the leader is still alive
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(3));
            interval.tick().await;
            loop {
                interval.tick().await;

                while !state.check_leader().await {
                    // try next leader in the LEADER_HOSTNAMES list
                    let mut leader_hostname = state.leader_hostname.lock().unwrap();
                    let index = LEADER_HOSTNAMES.iter().position(|&hostname| hostname == leader_hostname.as_str()).unwrap();
                    if index + 1 < LEADER_HOSTNAMES.len() {
                        *leader_hostname = LEADER_HOSTNAMES[index + 1].to_owned();
                    } else {
                        break;
                    }
                }
            }
        }});

        Ok(state)
    }

    pub fn storage_filename(filename: &str, version: i32) -> String {
        sanitize_filename::sanitize(format!("v{}.", version) + filename)
    }

    /// Merge the result of get_versions into one file with proper delimiters.
    pub async fn merge_versions(path: &Path, versions: impl IntoIterator<Item=i32>) -> io::Result<()> {
        let filename = path.file_name().unwrap().to_str().unwrap();
        let mut file = File::create(path).await?;

        let mut sorted_versions = versions.into_iter().collect::<Vec<i32>>();
        sorted_versions.sort_unstable();
        for version in sorted_versions.into_iter().rev() {
            // print delimiter with version to file
            file.write_all(format!("{:=^40}\n", format!(" Version {} ", version)).as_bytes()).await?;
            let mut version_file = File::open(path.with_file_name(format!("v{}.{}", version, filename))).await?;
            io::copy(&mut version_file, &mut file).await?;
            file.write_all(b"\n").await?;
        }
        Ok(())
    }

    pub fn leader_hostname(&self) -> String {
        self.leader_hostname.lock().unwrap().clone()
    }

    pub async fn check_leader(&self) -> bool {
        let leader_hostname = self.leader_hostname();
        let Ok(leader_client) = LeaderClient::spawn(&leader_hostname).await else { return false; };
        let Ok(result) = leader_client.alive(context::current()).await else { return false; };
        result
    }
}

impl MemberClient {
    pub async fn spawn(hostname: &str) -> io::Result<Self> {
        let transport = tarpc::serde_transport::tcp::connect((hostname, MEMBER_PORT), Json::default).await?;
        Ok(MemberClient::new(client::Config::default(), transport).spawn())
    }
}
