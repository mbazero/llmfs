pub struct Splitter {
    delimiters: Vec<String>,
}

impl Splitter {
    const DEFAULT_CHAR_DELIMS: &str = r#",.:;?_!"()' "#;
    const DEFAULT_STR_DELIMS: [&str; 1] = ["--"];

    pub fn new(delimiters: impl IntoIterator<Item = String>) -> Self {
        Self {
            delimiters: delimiters.into_iter().collect(),
        }
    }

    pub fn split<'a: 'b, 'b>(&'a self, text: &'b str) -> impl Iterator<Item = &'b str> {
        struct Iter<'a> {
            delimiters: &'a [String],
            next_delim: Option<&'a str>,
            remaining: &'a str,
        }

        impl<'a> Iterator for Iter<'a> {
            type Item = &'a str;

            fn next(&mut self) -> Option<Self::Item> {
                if let Some(next_delim) = self.next_delim.take() {
                    return Some(next_delim);
                }

                if self.remaining.is_empty() {
                    return None;
                }

                for (i, _) in self.remaining.char_indices() {
                    for d in self.delimiters {
                        if self.remaining[i..].starts_with(d) {
                            let d_end = i + d.len();
                            let token = &self.remaining[..i];
                            self.next_delim = Some(&self.remaining[i..d_end]);
                            self.remaining = &self.remaining[d_end..];
                            return Some(token);
                        }
                    }
                }

                Some(self.remaining)
            }
        }

        Iter {
            delimiters: &self.delimiters,
            next_delim: None,
            remaining: text,
        }
    }

    pub fn split_filter<'a: 'b, 'b>(&'a self, text: &'b str) -> impl Iterator<Item = &'b str> {
        self.split(text).filter_map(|s| {
            let s = s.trim();
            if s.is_empty() { None } else { Some(s) }
        })
    }
}

impl Default for Splitter {
    fn default() -> Self {
        let char_delims = Self::DEFAULT_CHAR_DELIMS.chars().map(|c| c.to_string());
        let str_delims = Self::DEFAULT_STR_DELIMS.into_iter().map(|s| s.to_string());
        Self::new(char_delims.chain(str_delims))
    }
}

#[cfg(test)]
mod tests {
    use crate::splitter::Splitter;

    #[test]
    fn test_split_filter() {
        let cases = [(
            "Hello, world. Is this-- a test?",
            vec![
                "Hello", ",", "world", ".", "Is", "this", "--", "a", "test", "?",
            ],
        )];

        let splitter = Splitter::default();

        for (i, (input, exp)) in cases.into_iter().enumerate() {
            let act: Vec<_> = splitter.split_filter(input).collect();
            assert_eq!(exp, act, "case {i} failed");
        }
    }
}
