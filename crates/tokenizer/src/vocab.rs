use indexmap::IndexSet;
use itertools::Itertools;

use crate::splitter::Splitter;

const EOT_TOKEN: &str = "<|endoftext|>";
const UNK_TOKEN: &str = "<|unk|>";

pub struct Vocab<'a> {
    mapping: IndexSet<&'a str>,
    unk_id: usize,
}

impl<'a> Vocab<'a> {
    pub fn from_tokens(tokens: impl IntoIterator<Item = &'a str>) -> Self {
        let mapping: IndexSet<_> = tokens
            .into_iter()
            .unique()
            .sorted()
            .chain([EOT_TOKEN, UNK_TOKEN])
            .collect();
        let unk_id = mapping.get_index_of(UNK_TOKEN).unwrap();
        Self { mapping, unk_id }
    }

    pub fn from_corpus(corpus: &'a str, splitter: &Splitter) -> Self {
        Self::from_tokens(splitter.split_filter(corpus))
    }

    pub fn get_token(&self, id: usize) -> Option<&'a str> {
        self.mapping.get_index(id).copied()
    }

    pub fn get_id(&self, token: &str) -> usize {
        self.mapping.get_index_of(token).unwrap_or(self.unk_id)
    }
}
