use std::collections::BTreeMap;
use std::ops::Bound::{Excluded, Unbounded};
use tabled::Tabled;

pub fn symmetric_ring_neighbors<'a, K, V, P>(map: &'a BTreeMap<K, V>, key: &K, k: i32, predicate: P) -> Vec<(&'a K, &'a V)>
    where K: Ord, P: Fn(&(&K, &V)) -> bool
{
    let mut neighbors = Vec::new();
    let mut left_iter = map.range((Unbounded, Excluded(key))).filter(&predicate);
    let mut right_iter = map.range((Excluded(key), Unbounded)).filter(&predicate);
    for _ in 0..k {
        left_iter.next_back()
            .or_else(|| right_iter.next_back())
            .map(|(k, v)| neighbors.push((k, v)));

        right_iter.next()
            .or_else(|| left_iter.next())
            .map(|(k, v)| neighbors.push((k, v)));
    }
    neighbors
}

#[derive(Tabled)]
pub struct Filename(#[tabled(rename = "filename")] pub String);

#[derive(Tabled)]
pub struct LatestVersion(#[tabled(rename = "latest_version")] pub i32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_ring_neighbors() {
        let mut map = BTreeMap::new();
        map.insert(1, "a");
        map.insert(2, "b");
        map.insert(3, "c");
        map.insert(4, "d");
        map.insert(5, "e");
        map.insert(6, "f");
        map.insert(7, "g");
        map.insert(8, "h");
        map.insert(9, "i");
        map.insert(10, "j");
        map.insert(11, "k");
        map.insert(12, "l");
        map.insert(13, "m");
        map.insert(14, "n");
        map.insert(15, "o");
        map.insert(16, "p");
        map.insert(17, "q");
        map.insert(18, "r");
        map.insert(19, "s");
        map.insert(20, "t");
        map.insert(21, "u");
        map.insert(22, "v");
        map.insert(23, "w");
        map.insert(24, "x");
        map.insert(25, "y");
        map.insert(26, "z");

        let neighbors = symmetric_ring_neighbors(&map, &13, 3, |_| true);
        assert_eq!(neighbors, vec![(&12, &"l"), (&14, &"n"), (&11, &"k"), (&15, &"o"), (&10, &"j"), (&16, &"p")]);
    }

    #[test]
    fn test_wrapped_ring_neighbors() {
        let mut map = BTreeMap::new();
        map.insert(1, "a");
        map.insert(2, "b");
        map.insert(3, "c");
        map.insert(4, "d");
        map.insert(5, "e");
        map.insert(6, "f");
        map.insert(7, "g");
        map.insert(8, "h");
        let neighbors = symmetric_ring_neighbors(&map, &8, 3, |_| true);
        assert_eq!(neighbors, vec![(&7, &"g"),(&1, &"a"), (&6, &"f"), (&2, &"b"), (&5, &"e"), (&3, &"c")]);
    }

    #[test]
    fn test_wrapped_overlap_ring_neighbors() {
        let mut map = BTreeMap::new();
        map.insert(1, "a");
        map.insert(2, "b");
        map.insert(3, "c");

        let neighbors = symmetric_ring_neighbors(&map, &2, 3, |_| true);
        assert_eq!(neighbors, vec![(&1, &"a"), (&3, &"c")]);
    }
}
