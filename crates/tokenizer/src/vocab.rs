use indexmap::IndexSet;
use itertools::Itertools;

use crate::splitter::Splitter;

pub struct Vocab<'a>(IndexSet<&'a str>);

impl<'a> Vocab<'a> {
    pub fn from_tokens(tokens: impl IntoIterator<Item = &'a str>) -> Self {
        Self(tokens.into_iter().unique().sorted().collect())
    }

    pub fn from_corpus(corpus: &'a str, splitter: &Splitter) -> Self {
        Self::from_tokens(splitter.split_filter(corpus))
    }

    pub fn get_token(&self, id: usize) -> Option<&'a str> {
        self.0.get_index(id).copied()
    }

    pub fn get_id(&self, token: &str) -> Option<usize> {
        self.0.get_index_of(token)
    }
}
