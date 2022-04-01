use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub struct AutoDiffError {
    details: String,
}

impl AutoDiffError {
    pub fn new(msg: &str) -> AutoDiffError {
        AutoDiffError {
            details: msg.to_string(),
        }
    }
}

impl fmt::Display for AutoDiffError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.details)
    }
}

impl Error for AutoDiffError {
    fn description(&self) -> &str {
        &self.details
    }
}

impl From<AutoDiffError> for std::fmt::Error {
    fn from(item: AutoDiffError) -> std::fmt::Error {
        std::fmt::Error::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        fn return_err() -> Result<(), AutoDiffError> {
            Err(AutoDiffError::new(&format!("{:?}", 12)))
        }

        let e = return_err();
        assert!(e.is_err());
    }
}
