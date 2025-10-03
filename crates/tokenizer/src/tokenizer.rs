use anyhow::Result;

use bimap::BiHashMap;
use regex::Regex;

pub trait Tokenizer {
    fn new(vocab: impl IntoIterator<Item = (usize, String)>) -> Result<Self>
    where
        Self: Sized;

    fn encode(&mut self, text: &str) -> impl Iterator<Item = usize>;

    fn decode(&mut self, ids: impl IntoIterator<Item = usize>) -> impl Iterator<Item = &str>;
}

pub struct BasicTokenizer {
    vocab: BiHashMap<usize, String>,
    re: Regex,
}

impl Tokenizer for BasicTokenizer {
    fn new(vocab: impl IntoIterator<Item = (usize, String)>) -> Result<Self> {
        Ok(Self {
            vocab: vocab.into_iter().collect(),
            re: Regex::new(r#"([,.:;?_!"()\']|--|\s)"#)?,
        })
    }

    fn encode(&mut self, text: &str) -> impl Iterator<Item = usize> {
        self.re.split(text).filter_map(|s| {
            let s = s.trim();

            if s.is_empty() {
                return None;
            }

            let Some(&id) = self.vocab.get_by_right(s) else {
                panic!("token {s} is not present in the vocabulary");
            };

            Some(id)
        })
    }

    fn decode(&mut self, ids: impl IntoIterator<Item = usize>) -> impl Iterator<Item = &str> {
        ids.into_iter().map(|i| {
            let Some(s) = self.vocab.get_by_left(&i) else {
                panic!("token id {i} is not present in the vocabulary");
            };
            s.as_str()
        })
    }
}
