#[cfg(feature = "use-serde")]
use serde::{
    de, de::MapAccess, de::SeqAccess, de::Visitor, ser::SerializeStruct, Deserialize, Deserializer,
    Serialize, Serializer,
};
use std::fmt;

use crate::var::Var;

impl Serialize for Var {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // 3 is the number of fields in the struct.
        let mut state = serializer.serialize_struct("Var", 1)?;
        state.serialize_field("var", &*self.inner().borrow())?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Var {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field {
            Var,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("var")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "var" => Ok(Field::Var),
                            _ => Err(de::Error::unknown_field(value, &FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct VarVisitor;

        impl<'de> Visitor<'de> for VarVisitor {
            type Value = Var;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Var")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Var, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut var = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Var => {
                            if var.is_some() {
                                return Err(de::Error::duplicate_field("var"));
                            }
                            var = Some(map.next_value()?);
                        }
                    }
                }
                let var = var.ok_or_else(|| de::Error::missing_field("id"))?;
                Ok(Var::set_inner(var))
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Var, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let var = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                Ok(Var::set_inner(var))
            }
        }

        const FIELDS: [&str; 1] = ["var"];
        deserializer.deserialize_struct("Duration", &FIELDS, VarVisitor)
    }
}

#[cfg(all(test, feature = "use-serde"))]
mod tests {
    use crate::var::Var;
    use rand::prelude::*;

    #[test]
    fn test_serde_var_inner() {
        let mut rng = StdRng::seed_from_u64(671);
        let n = 10;
        let data = Var::normal(&mut rng, &vec![n, 2], 0., 2.);
        let result = data.matmul(&Var::new(&vec![2., 3.], &vec![2, 1])).unwrap()
            + Var::new(&vec![1.], &vec![1]);

        let serialized = serde_pickle::to_vec(&result, true).unwrap();
        let deserialized: Var = serde_pickle::from_slice(&serialized).unwrap();
        println!("{:?}", deserialized.dump_net());
        assert_eq!(result, deserialized);
    }
}
