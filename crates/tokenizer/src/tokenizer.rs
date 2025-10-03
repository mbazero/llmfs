use crate::{splitter::Splitter, vocab::Vocab};

pub struct Tokenizer<'a> {
    vocab: Vocab<'a>,
    splitter: Splitter,
}

impl<'a> Tokenizer<'a> {
    pub fn from_corpus(corpus: &'a str, splitter: Splitter) -> Self {
        Self {
            vocab: Vocab::from_corpus(corpus, &splitter),
            splitter,
        }
    }

    pub fn encode(&self, text: &str) -> impl Iterator<Item = usize> {
        self.splitter
            .split_filter(text)
            .map(|s| match self.vocab.get_id(s) {
                Some(id) => id,
                None => panic!("token {s} is not present in the vocabulary"),
            })
    }

    // FIXME!
    pub fn decode(&self, ids: impl IntoIterator<Item = usize>) -> impl Iterator<Item = &str> {
        ids.into_iter().map(|i| match self.vocab.get_token(i) {
            Some(t) => t,
            None => panic!("token id {i} is not present in the vocabulary"),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{splitter::Splitter, tokenizer::Tokenizer, util::read_corpus};
    use anyhow::Result;
    use itertools::Itertools;

    const TEXT: &str =
        r#""It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."#;

    const IDS: [usize; 21] = [
        1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7,
    ];

    #[test]
    fn test_encode() -> Result<()> {
        let corpus = read_corpus()?;
        let tokenizer = Tokenizer::from_corpus(&corpus, Splitter::default());
        let encoded_ids: Vec<_> = tokenizer.encode(TEXT).collect();
        assert_eq!(IDS, encoded_ids.as_slice(), "encoded ids don't match");
        Ok(())
    }

    #[test]
    fn test_decode() -> Result<()> {
        const EXP_TEXT: &str = r#"" It ' s the last he painted , you know , " Mrs. Gisburn said with pardonable pride."#;
        let corpus = read_corpus()?;
        let tokenizer = Tokenizer::from_corpus(&corpus, Splitter::default());
        let decoded_text = tokenizer.decode(IDS).join(" ");
        assert_eq!(EXP_TEXT, decoded_text, "decoded text doesn't match");
        Ok(())
    }
}
