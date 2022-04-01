#[cfg(feature = "use-serde")]
use serde::{
    de, de::MapAccess, de::SeqAccess, de::Visitor, ser::SerializeStruct, Deserialize, Deserializer,
    Serialize, Serializer,
};
use std::fmt;

use crate::var_inner::VarInner;

impl Serialize for VarInner {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // 3 is the number of fields in the struct.
        let mut state = serializer.serialize_struct("VarInner", 3)?;
        state.serialize_field("id", &self.get_id())?;
        state.serialize_field("need_grad", &self.get_need_grad())?;
        state.serialize_field("net", &*self.get_net().borrow())?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for VarInner {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field {
            Id,
            NeedGrad,
            Net,
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
                        formatter.write_str("id, need_grad, or net")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "id" => Ok(Field::Id),
                            "need_grad" => Ok(Field::NeedGrad),
                            "net" => Ok(Field::Net),
                            _ => Err(de::Error::unknown_field(value, &FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct VarInnerVisitor;

        impl<'de> Visitor<'de> for VarInnerVisitor {
            type Value = VarInner;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct VarInner")
            }

            fn visit_map<V>(self, mut map: V) -> Result<VarInner, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut id = None;
                let mut need_grad = None;
                let mut net = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Id => {
                            if id.is_some() {
                                return Err(de::Error::duplicate_field("id"));
                            }
                            id = Some(map.next_value()?);
                        }
                        Field::NeedGrad => {
                            if need_grad.is_some() {
                                return Err(de::Error::duplicate_field("need_grad"));
                            }
                            need_grad = Some(map.next_value()?);
                        }
                        Field::Net => {
                            if net.is_some() {
                                return Err(de::Error::duplicate_field("net"));
                            }
                            net = Some(map.next_value()?);
                        }
                    }
                }
                let id = id.ok_or_else(|| de::Error::missing_field("id"))?;
                let need_grad = need_grad.ok_or_else(|| de::Error::missing_field("need_grad"))?;
                let net = net.ok_or_else(|| de::Error::missing_field("net"))?;
                Ok(VarInner::set_inner(id, need_grad, net))
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<VarInner, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let id = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let need_grad = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let net = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                Ok(VarInner::set_inner(id, need_grad, net))
            }
        }

        const FIELDS: [&str; 3] = ["id", "need_grad", "net"];
        deserializer.deserialize_struct("Duration", &FIELDS, VarInnerVisitor)
    }
}

#[cfg(all(test, feature = "use-serde"))]
mod tests {
    use crate::var::Var;
    use crate::var_inner::VarInner;
    use rand::prelude::*;

    #[test]
    fn test_serde_var_inner() {
        let mut rng = StdRng::seed_from_u64(671);
        let n = 10;
        let data = Var::normal(&mut rng, &vec![n, 2], 0., 2.);
        let result = data.matmul(&Var::new(&vec![2., 3.], &vec![2, 1])).unwrap()
            + Var::new(&vec![1.], &vec![1]);

        let serialized = serde_pickle::to_vec(&*result.inner().borrow(), true).unwrap();
        let deserialized: VarInner = serde_pickle::from_slice(&serialized).unwrap();
        println!("{:?}", deserialized.dump_net());
        //assert_eq!(*result.dump_net().borrow(), deserialized);
    }
}
