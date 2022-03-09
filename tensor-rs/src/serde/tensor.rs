#[cfg(feature = "use-serde")]
use serde::{Serialize, Deserialize, Serializer, Deserializer,
	    ser::SerializeStruct,
	    de, de::Visitor, de::SeqAccess, de::MapAccess};
use crate::tensor::Tensor;
use std::fmt;

impl Serialize for Tensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer, {
        // 3 is the number of fields in the struct.
        let mut state = serializer.serialize_struct("Tensor", 3)?;
        state.serialize_field("v", &self.inner().borrow().clone())?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Tensor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de>, {

	enum Field { V }
	
        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
            where D: Deserializer<'de>, {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("v")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where E: de::Error, {
                        match value {
                            "v" => Ok(Field::V),
                            _ => Err(de::Error::unknown_field(value, &FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }
	
        struct TensorVisitor;

        impl<'de> Visitor<'de> for TensorVisitor {
            type Value = Tensor;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Tensor")
            }

	    fn visit_map<V>(self, mut map: V) -> Result<Tensor, V::Error>
            where V: MapAccess<'de>, {
		let mut v = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::V => {
                            if v.is_some() {
                                return Err(de::Error::duplicate_field("v"));
                            }
                            v = Some(map.next_value()?);
                        }
                    }
                }
                let v = v.ok_or_else(|| de::Error::missing_field("ok"))?;
                Ok(Tensor::set_inner(v))
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Tensor, V::Error>
            where V: SeqAccess<'de>, {
                let tt = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                Ok(Tensor::set_inner(tt))
            }
        }

        const FIELDS: [&str; 1] = ["v"];
        deserializer.deserialize_struct("Duration", &FIELDS, TensorVisitor)
    }
}


#[cfg(all(test, feature = "use-serde"))]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn test_serde() {
	let m1 = Tensor::eye(3,3);

        let serialized = serde_pickle::to_vec(&m1, true).unwrap();
        let deserialized = serde_pickle::from_slice(&serialized).unwrap();
        //println!("{:?}", deserialized);
        assert_eq!(m1, deserialized);
    }
}
