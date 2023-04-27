mod membership;
mod utils;
mod services;
mod model_utils;

use std::collections::HashSet;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use enclose::enclose;
use futures::prelude::*;
use histogram::Histogram;
use log::{LevelFilter, info, warn};
use nix::unistd::gethostname;
use tarpc::{client, context};
use tarpc::server::{self, incoming::Incoming, Channel, Serve};
use tarpc::tokio_serde::formats::Json;
use path_absolutize::*;
use tabled::Table;
use membership::*;
use services::*;
use utils::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let hostname = gethostname().unwrap().into_string().unwrap();
    simple_logging::log_to_file(hostname.clone() + ".log", LevelFilter::Info).unwrap();
    let membership_service = Arc::new(MembershipService::run());

    let member_server = MemberState::shared(membership_service.clone()).await?;
    tokio::spawn(serve_member(member_server.clone(), membership_service.clone()));

    if LEADER_HOSTNAMES.contains(&hostname.as_str()) {
        tokio::spawn(serve_leader(membership_service.clone(), member_server.clone()));
    }

    run_cli(member_server.clone(), membership_service.clone())?;

    Ok(())
}

async fn serve_leader(membership_service: Arc<MembershipService>, member_state: Arc<MemberState>) -> anyhow::Result<()> {
    let leader_server = LeaderState::new(membership_service.clone(), member_state).await;

    let server_addr = ("0.0.0.0", LEADER_PORT);
    let mut listener = tarpc::serde_transport::tcp::listen(&server_addr, Json::default).await?;
    listener.config_mut().max_frame_length(usize::MAX);
    listener
        // ignore accept errors
        .filter_map(|r| async move { r.ok() })
        .map(server::BaseChannel::with_defaults)
        .max_concurrent_requests_per_channel(1)
        .map(enclose! { (leader_server) move |channel| {
            // one channel for each client
            info!("Leader: New client: {}", channel.transport().peer_addr().unwrap());
            // Channel::execute(self, serve: impl Serve) -> impl Future<Output = ()>
            channel.execute(leader_server.clone().serve())
        }})
        // run the futures with buffering
        .buffer_unordered(10)
        .for_each(|_| async move {})
        .await;
    Ok(())
}

async fn serve_member(member_server: MemberServer, membership_service: Arc<MembershipService>) -> anyhow::Result<()> {
    let server_addr = ("0.0.0.0", MEMBER_PORT);
    let mut listener = tarpc::serde_transport::tcp::listen(&server_addr, Json::default).await?;
    listener.config_mut().max_frame_length(usize::MAX);
    listener
        .filter_map(|r| async move { r.ok() })
        .map(server::BaseChannel::with_defaults)
        .max_concurrent_requests_per_channel(1)
        .map(enclose! { (member_server) move |channel| {
            info!("Member: New client: {}", channel.transport().peer_addr().unwrap());
            channel.execute(member_server.clone().serve())
        }})
        .buffer_unordered(10)
        .for_each(|_| async move {})
        .await;
    Ok(())
}

fn run_cli(member_server: MemberServer, membership_service: Arc<MembershipService>) -> io::Result<()> {
    for line in io::stdin().lines() {
        let line = line?;
        let tokens: Vec<&str> = line.split_whitespace().collect();

        if tokens.is_empty() {
            eprintln!("Invalid command!");
            continue;
        }

        match tokens[0] {
            // membership
            "list_mem" | "lm" => {
                membership_service.list_membership();
            }
            "list_self" => {
                membership_service.list_self();
            }
            "join" | "j" => {
                if tokens.len() != 2 {
                    eprintln!("Invalid join command!");
                    continue;
                }
                membership_service.join(tokens[1])?;
            }
            "leave" | "l" => {
                membership_service.leave();
            }
            // file system
            "put" | "p" => {
                if tokens.len() != 3 {
                    eprintln!("Invalid put command!");
                    continue;
                }
                // absolutize the path since scp doesn't run locally
                let src_path = match Path::new(tokens[1]).absolutize() {
                    Ok(path) => PathBuf::from(path),
                    Err(e) => {
                        eprintln!("Invalid local path: {}", e);
                        continue;
                    }
                };
                let filename = tokens[2].to_owned();
                tokio::spawn(enclose! { (member_server, membership_service) async move {
                    let leader = LeaderClient::spawn(&member_server.leader_hostname()).await?;
                    // putting can take a long time, so we extend the timeout
                    let mut ctx = context::current();
                    ctx.deadline = SystemTime::now() + Duration::from_secs(3600);
                    let received_ids = leader.put(ctx, membership_service.id(), src_path, filename).await?;
                    println!("Stored on:\n{}", Table::new(received_ids));
                    Ok::<(), anyhow::Error>(())
                }});
            }
            "get" | "g" => {
                if tokens.len() != 3 {
                    eprintln!("Invalid get command!");
                    continue;
                }
                let filename = tokens[1].to_owned();
                let dest_path = match Path::new(tokens[2]).absolutize() {
                    Ok(path) => PathBuf::from(path),
                    Err(e) => {
                        eprintln!("Invalid local path: {}", e);
                        continue;
                    }
                };
                tokio::spawn(enclose! { (member_server, membership_service) async move {
                    let leader = LeaderClient::spawn(&member_server.leader_hostname()).await?;
                    let mut ctx = context::current();
                    ctx.deadline = SystemTime::now() + Duration::from_secs(3600);
                    let result: Option<i32> = leader.get(ctx, filename, membership_service.id(), dest_path).await?;
                    match result {
                        Some(version) => println!("Retrieved version: {}", version),
                        None => println!("File not found!"),
                    }
                    Ok::<(), anyhow::Error>(())
                }});
            }
            "delete" | "d" => {
                if tokens.len() != 2 {
                    eprintln!("Invalid delete command!");
                    continue;
                }
                let filename = tokens[1].to_owned();
                tokio::spawn(enclose! { (member_server) async move {
                    let leader = LeaderClient::spawn(&member_server.leader_hostname()).await?;
                    leader.delete(context::current(), filename).await?;
                    println!("Deleted!");
                    Ok::<(), anyhow::Error>(())
                }});
            }
            "ls" => {
                if tokens.len() != 2 {
                    eprintln!("Invalid ls command!");
                    continue;
                }
                let filename = tokens[1].to_owned();
                tokio::spawn(enclose! { (member_server) async move {
                    let leader = LeaderClient::spawn(&member_server.leader_hostname()).await?;
                    let map: Vec<(Id, Vec<i32>)> = leader.ls(context::current(), filename).await?;
                    let locations = map.iter().take(4).map(|(id, versions)| (id, LatestVersion(*versions.iter().max().unwrap_or(&0))));
                    println!("{}", Table::new(locations));
                    Ok::<(), anyhow::Error>(())
                }});
            }
            "store" | "s" => {
                if tokens.len() != 1 {
                    eprintln!("Invalid store command!");
                    continue;
                }
                let table = Table::new(
                    member_server.files.lock().unwrap()
                        .clone()
                        .into_iter()
                        .map(|(filename, versions)| (Filename(filename), LatestVersion(*versions.iter().max().unwrap_or(&0))))
                );
                println!("{}", table);
            }
            "get-versions" | "gv" => {
                if tokens.len() != 4 {
                    eprintln!("Invalid get-versions command!");
                    continue;
                }
                let filename = tokens[1].to_owned();
                let count = match tokens[2].parse::<i32>() {
                    Ok(count) => count,
                    Err(e) => {
                        eprintln!("Invalid count: {}", e);
                        continue;
                    }
                };
                let dest_path = match Path::new(tokens[3]).absolutize() {
                    Ok(path) => PathBuf::from(path),
                    Err(e) => {
                        eprintln!("Invalid local path: {}", e);
                        continue;
                    }
                };
                tokio::spawn(enclose! { (member_server, membership_service) async move {
                    let leader = LeaderClient::spawn(&member_server.leader_hostname()).await?;
                    let versions: HashSet<i32> = leader.get_versions(context::current(), filename, count, membership_service.id(), dest_path.clone()).await?;
                    MemberState::merge_versions(&dest_path, versions.clone()).await?;
                    println!("Retrieved versions: {:?}", versions);
                    Ok::<(), anyhow::Error>(())
                }});
            }
            "train" | "t" => {
                if tokens.len() != 3 {
                    eprintln!("Invalid train command!");
                    continue;
                }
                // let every vm load the file
                println!("Starting training...");
                let filename = tokens[1].to_owned();
                let dest_path = match Path::new(&format!("{}.ot", tokens[2])).absolutize() {
                    Ok(path) => PathBuf::from(path),
                    Err(e) => {
                        eprintln!("Invalid local path: {}", e);
                        continue;
                    }
                };
                tokio::spawn(enclose! { (member_server, membership_service) async move {
                    let leader = LeaderClient::spawn(&member_server.leader_hostname()).await?;
                    let mut ctx = context::current();
                    ctx.deadline = SystemTime::now() + Duration::from_secs(3600);
                    match leader.train(ctx, filename, dest_path).await {
                        Ok(()) => println!("Training complete!"),
                        Err(e) => eprintln!("Training failed: {}", e),
                    }
                    Ok::<(), anyhow::Error>(())
                }});
            }
            "predict" => {
                if tokens.len() != 1 {
                    eprintln!("Invalid train command!");
                    continue;
                }

                tokio::spawn(enclose! { (member_server, membership_service) async move {
                    let leader = LeaderClient::spawn(&member_server.leader_hostname()).await?;
                    let mut ctx = context::current();
                    ctx.deadline = SystemTime::now() + Duration::from_secs(3600);
                    leader.predict(ctx, "".to_owned(), "".to_owned()).await?;
                    Ok::<(), anyhow::Error>(())
                }});
            }
            "jobs" => {
                if tokens.len() != 1 {
                    eprintln!("Invalid jobs command!");
                    continue;
                }

                tokio::spawn(enclose! { (member_server, membership_service) async move {
                    let leader = LeaderClient::spawn(&member_server.leader_hostname()).await?;
                    let jobs: Vec<Job> = leader.jobs(context::current()).await?;

                    for (i, job) in jobs.into_iter().enumerate() {
                        let mut query_durations = Histogram::new();
                        for duration in &job.query_durations {
                            _ = query_durations.increment(duration.as_millis() as u64);
                        }
                        let accuracy = if job.finished_prediction_count > 0 {
                            job.correct_prediction_count as f64 / job.finished_prediction_count as f64
                        } else {
                            0.0
                        };
                        println!(
                            "Job {n}:\n\
                            \tModel: {model_name}\n\
                            \tAccuracy: {correct}/{finished} = {accuracy:.2}%\n\
                            \tQueries: {query_count} total, {query_mean} ms avg, {query_std} ms std, {query_median} ms median,\n\
                            \t\t{query_p90} ms p90, {query_p95} ms p95, {query_p99} ms p99",
                            n=i + 1,
                            model_name=job.model_name,
                            correct=job.correct_prediction_count,
                            finished=job.finished_prediction_count,
                            accuracy=accuracy * 100.0,
                            query_count=job.query_durations.len(),
                            query_mean=query_durations.mean().unwrap_or(0),
                            query_std=query_durations.stddev().unwrap_or(0),
                            query_median=query_durations.percentile(50.0).unwrap_or(0),
                            query_p90=query_durations.percentile(90.0).unwrap_or(0),
                            query_p95=query_durations.percentile(95.0).unwrap_or(0),
                            query_p99=query_durations.percentile(99.0).unwrap_or(0)
                        );
                    }

                    Ok::<(), anyhow::Error>(())
                }});
            }
            "assign" => {
                if tokens.len() != 1 {
                    eprintln!("Invalid assign command!");
                    continue;
                }

                tokio::spawn(enclose! { (member_server, membership_service) async move {
                    let leader = LeaderClient::spawn(&member_server.leader_hostname()).await?;
                    let jobs: Vec<Job> = leader.jobs(context::current()).await?;

                    for (i, job) in jobs.into_iter().enumerate() {
                        println!("Job {}:\n{}", i + 1, Table::new(job.assigned_member_ids));
                    }

                    Ok::<(), anyhow::Error>(())
                }});
            }
            _ => {
                warn!("Unknown command");
            }
        }
    }
    Ok(())
}
