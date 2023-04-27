use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::{fmt, io, thread};
use std::fmt::Debug;
use std::net::{ToSocketAddrs, UdpSocket};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use nix::unistd::gethostname;
use serde::{Serialize, Deserialize};
use chrono::prelude::*;
use serde_with::serde_as;
use tabled::{Table, Tabled};
use crate::utils::symmetric_ring_neighbors;
use log::{info, warn};
use enclose::enclose;

pub type MembershipList = BTreeMap<Id, Membership>;

/// The service is NOT terminated when the value is dropped.
#[derive(Debug)]
pub struct MembershipService {
    pub membership_list: Arc<Mutex<MembershipList>>,
    pub id: Arc<Mutex<Id>>,
}

#[derive(Serialize, Deserialize, Tabled, Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Clone)]
pub struct Id {
    pub address: String,
    pub timestamp: DateTime<Local>,
}

#[derive(Serialize, Deserialize, Tabled, Debug, Eq, PartialEq, Clone)]
pub struct Membership {
    pub status: Status,
    last_active: DateTime<Local>,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq, Clone, Copy)]
pub enum Status {
    Active,
    Failed,
}

#[serde_as]
#[derive(Serialize, Deserialize, Debug)]
enum Message {
    Ping {
        sender_id: Id,
        #[serde_as(as = "Vec<(_, _)>")]
        membership_list: MembershipList,
    },
    Ack {
        sender_id: Id,
        last_active: DateTime<Local>,
    },
    Join { sender_id: Id },
    Welcome {
        sender_id: Id,
        #[serde_as(as = "Vec<(_, _)>")]
        membership_list: MembershipList,
    },
}

const RECEIVER_PORT: u16 = 8850;

impl MembershipService {
    pub fn run() -> Self {
        let hostname = gethostname().unwrap().into_string().unwrap();
        let address = format!("{}:{}", hostname, RECEIVER_PORT);
        println!("Address is {}", address);

        let membership_list = Arc::new(Mutex::new(MembershipList::new()));
        let id = Arc::new(Mutex::new(Id {
            address,
            timestamp: Local::now(),
        }));

        thread::spawn(enclose! { (membership_list, id) move || {
            if let Err(error) = run_receiver(membership_list, id) {
                println!("Receiver error: {}", error);
            }
        }});

        thread::spawn(enclose! { (membership_list, id) move || {
            if let Err(error) = run_pinger(membership_list, id) {
                println!("Pinger error: {}", error);
            }
        }});

        thread::spawn(enclose! { (membership_list, id) move || {
            run_detector(membership_list, id);
        }});

        Self {
            membership_list,
            id,
        }
    }

    pub fn list_membership(&self) {
        let membership_list = self.membership_list.lock().unwrap();
        let active_membership_list = membership_list.iter().filter(|(_, m)| m.status == Status::Active);
        println!("{}", Table::new(active_membership_list));
    }

    pub fn list_self(&self) {
        let id = self.id.lock().unwrap();
        let message = format!("ID: {:?}", &id);
        info!("{}",message);
        println!("{}", message);
    }

    pub fn join(&self, hostname: &str) -> io::Result<()> {
        let id = {
            let mut id = self.id.lock().unwrap();
            id.timestamp = Local::now();
            id.clone()
        };
        let message = Message::Join { sender_id: id };
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        send_message(&socket, (hostname, RECEIVER_PORT), &message);
        Ok(())
    }

    pub fn leave(&self) {
        let mut membership_list = self.membership_list.lock().unwrap();

        let message = format!("Leaving group...\nLast membership list:\n{}", Table::new(&membership_list as &MembershipList));
        info!("{}",message);
        println!("{}", message);
        membership_list.clear();
    }
}

impl MembershipService {
    pub fn id(&self) -> Id {
        self.id.lock().unwrap().clone()
    }
    pub fn active_ids(&self) -> HashSet<Id> {
        self.membership_list
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, m)| m.status == Status::Active)
            .map(|(id, _)| id.clone())
            .collect()
    }
}

fn run_receiver(membership_list: Arc<Mutex<MembershipList>>, id: Arc<Mutex<Id>>) -> Result<(), Box<dyn Error>> {
    let socket = UdpSocket::bind(("0.0.0.0", RECEIVER_PORT))?;
    let mut buf = [0; 4096];
    loop {
        let (amt, src) = socket.recv_from(&mut buf)?;

        // deserialize message
        let message: Option<Message> = flexbuffers::Reader::get_root(&buf[..amt])
            .ok()
            .and_then(|r| Message::deserialize(r).ok());

        match message {
            Some(Message::Ping { sender_id, membership_list: remote_membership_list }) => {
                let in_group = {
                    let mut membership_list = membership_list.lock().unwrap();
                    update_membership_list(&mut membership_list, &remote_membership_list);
                    !membership_list.is_empty()  // membership list is empty when we haven't joined a group or have left the group
                };

                if in_group {
                    let id = id.lock().unwrap().clone();
                    let message = Message::Ack {
                        sender_id: id,
                        last_active: Local::now(),
                    };
                    send_message(&socket, sender_id.address, &message);
                }
            }
            Some(Message::Ack { sender_id, last_active }) => {
                let mut membership_list = membership_list.lock().unwrap();
                update_membership_list(
                    &mut membership_list,
                    [(&sender_id, &Membership { status: Status::Active, last_active })],
                );
            }
            Some(Message::Join { sender_id }) => {
                let id = id.lock().unwrap().clone();
                let membership_list = {
                    let mut membership_list = membership_list.lock().unwrap();

                    // fail all existing members with the same address to handle fast rejoins
                    membership_list.iter_mut()
                        .filter(|(id, _)| id.address == sender_id.address)
                        .for_each(|(_, m)| m.status = Status::Failed);

                    membership_list.insert(sender_id.clone(), Membership {
                        status: Status::Active,
                        last_active: sender_id.timestamp.clone(),
                    });

                    // update own membership before sending a message containing the membership list
                    membership_list.get_mut(&id)
                        .map(|m| {
                            m.status = Status::Active;
                            m.last_active = Local::now()
                        });

                    membership_list.clone()
                };
                let message = Message::Welcome {
                    sender_id: id,
                    membership_list,
                };
                send_message(&socket, sender_id.address, &message);
            }
            Some(Message::Welcome { sender_id: _, membership_list: remote_membership_list }) => {
                let mut membership_list = membership_list.lock().unwrap();
                *membership_list = remote_membership_list;
                println!("Joined!\nInitial membership list:\n{}", Table::new(&membership_list as &MembershipList));
            }
            None => {}
        }
    }
}

fn run_pinger(membership_list: Arc<Mutex<MembershipList>>, id: Arc<Mutex<Id>>) -> Result<(), io::Error> {
    let socket = UdpSocket::bind("0.0.0.0:0")?;
    // store a copy of active neighbors to detect changes
    let mut active_neighbors = Vec::new();
    loop {
        thread::sleep(Duration::from_secs(1));

        let id = id.lock().unwrap().clone();
        let membership_list = {
            let mut membership_list = membership_list.lock().unwrap();
            membership_list.get_mut(&id)
                .map(|m| {
                    m.status = Status::Active;
                    m.last_active = Local::now();
                });
            membership_list.clone()
        };
        let new_active_neighbors =
            symmetric_ring_neighbors(&membership_list, &id, 2, |(_, m)| m.status == Status::Active)
                .into_iter()
                .map(|(id, _)| id.clone())
                .collect::<Vec<_>>();
        if new_active_neighbors != active_neighbors {
            let message = format!("Active neighbors have changed:\n{}", Table::new(&new_active_neighbors));
            info!("{}", message);

            active_neighbors = new_active_neighbors;
        }

        let message = Message::Ping { sender_id: id, membership_list };
        for Id { address, .. } in &active_neighbors {
            send_message(&socket, address, &message);
        }
    }
}

fn run_detector(membership_list: Arc<Mutex<MembershipList>>, id: Arc<Mutex<Id>>) {
    let mut last_active_neighbors = Vec::new();
    loop {
        let id = id.lock().unwrap().clone();
        let mut membership_list = membership_list.lock().unwrap();
        let now = Local::now();

        // detect failures of active neighbors 1 second ago
        // to give time to ping new neighbors
        for i in &last_active_neighbors {
            if let Some(m @ Membership { status: Status::Active, .. }) = membership_list.get_mut(i) {
                let elapsed = now - m.last_active;
                if elapsed > chrono::Duration::seconds(3) {
                    warn!("Detected failure of {:?} that hasn't been updated for {}", i, elapsed);
                    println!("Detected failure of {:?} that hasn't been updated for {}", i, elapsed);
                    m.status = Status::Failed;
                }
            }
        }

        last_active_neighbors = symmetric_ring_neighbors(&membership_list, &id, 2, |(_, m)| m.status == Status::Active)
            .into_iter()
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();

        // unlock membership list mutex before sleeping
        drop(membership_list);

        thread::sleep(Duration::from_secs(1));
    }
}

fn send_message(socket: &UdpSocket, dest: impl ToSocketAddrs + Debug, message: &Message) {
    let mut writer = flexbuffers::FlexbufferSerializer::new();
    message.serialize(&mut writer).unwrap();
    match socket.send_to(writer.view(), &dest) {
        Ok(_) => {},
        Err(error) => warn!("Error sending message {:?}: {}", message, error),
    }
}

fn update_membership_list<'a>(membership_list: &mut MembershipList, remote_membership_list: impl IntoIterator<Item=(&'a Id, &'a Membership)>) {
    // may receive Ping/Ack after leaving
    if membership_list.is_empty() { return; }

    for (id, remote_membership) in remote_membership_list {
        if membership_list.contains_key(id) {
            let membership = membership_list.get_mut(id).unwrap();
            // when last_active is the same, Failed status has precedence
            if membership.last_active < remote_membership.last_active ||
                (membership.last_active == remote_membership.last_active && remote_membership.status == Status::Failed)
            {
                let message = format!("Updating membership for {}: {:?} -> {:?}",
                                      id.address.clone(),
                                      membership.status,
                                      remote_membership.status);
                if membership.status != remote_membership.status {
                    println!("{}", message);
                }
                info!("{}",message);
                *membership = remote_membership.clone();
            }
        } else {
            membership_list.insert(id.clone(), remote_membership.clone());
        }
    }
}

impl Message {
    fn sender_id(&self) -> &Id {
        match self {
            Message::Ping { sender_id, .. } => sender_id,
            Message::Ack { sender_id, .. } => sender_id,
            Message::Join { sender_id } => sender_id,
            Message::Welcome { sender_id, .. } => sender_id,
        }
    }
}

impl Id {
    pub fn hostname(&self) -> String {
        self.address.to_string().split(':').next().unwrap().to_owned()
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
