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
            .map(|s| self.vocab.get_id(s))
    }

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
        const EXP_TEXT: &str = r#"" It ' s the last he painted , you know , " Mrs . Gisburn said with pardonable pride ."#;
        let corpus = read_corpus()?;
        let tokenizer = Tokenizer::from_corpus(&corpus, Splitter::default());
        let decoded_text = tokenizer.decode(IDS).join(" ");
        assert_eq!(EXP_TEXT, decoded_text, "decoded text doesn't match");
        Ok(())
    }

    #[test]
    fn test_encode_decode_with_unc() -> Result<()> {
        const INPUT_TEXT: &str =
            r#"Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."#;
        const EXP_IDS: [usize; 16] = [
            1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7,
        ];
        const EXP_OUTPUT: &str =
            r#"<|unk|> , do you like tea ? <|endoftext|> In the sunlit terraces of the <|unk|> ."#;

        let corpus = read_corpus()?;
        let tokenizer = Tokenizer::from_corpus(&corpus, Splitter::default());
        let act_ids: Vec<_> = tokenizer.encode(INPUT_TEXT).collect();
        assert_eq!(EXP_IDS, act_ids.as_slice(), "encoded ids don't match");
        let act_output = tokenizer.decode(act_ids).join(" ");
        assert_eq!(EXP_OUTPUT, act_output, "output text doesn't match");
        Ok(())
    }
}
