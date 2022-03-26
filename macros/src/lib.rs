/// procedure macros

use proc_macro::TokenStream;
use syn::{parse_macro_input, ItemStruct, parse, parse::Parser};
use syn::punctuated::Punctuated;
use syn::{Expr, Token};
use quote::quote;


#[proc_macro_attribute]
pub fn add_op_handle(args: TokenStream, input: TokenStream) -> TokenStream {
    let mut item_struct = parse_macro_input!(input as ItemStruct);
    let _ = parse_macro_input!(args as parse::Nothing);

    if let syn::Fields::Named(ref mut fields) = item_struct.fields {

        fields.named.push(
            syn::Field::parse_named
                .parse2(quote! {
                    #[cfg_attr(feature = "use-serde", serde(skip))]
                    handle: OpHandle
                })
                .unwrap(),
        );
    }

    return quote! {
        #item_struct
    }
    .into();
}

#[proc_macro_attribute]
pub fn extend_op_impl(args: TokenStream, input: TokenStream) -> TokenStream {
    let mut item_struct = parse_macro_input!(input as ItemStruct);
    let _ = parse_macro_input!(args as parse::Nothing);

    if let syn::Fields::Named(ref mut fields) = item_struct.fields {

        fields.named.push(
            syn::Field::parse_named
                .parse2(quote! {
                    #[cfg_attr(feature = "use-serde", serde(skip))]
                    handle: OpHandle
                })
                .unwrap(),
        );
    }

    return quote! {
        #item_struct
    }
    .into();
}


#[proc_macro]
pub fn gen_serde_funcs(input: TokenStream) -> TokenStream {

    let input_tokens = input.clone();
    let parser = Punctuated::<Expr, Token![,]>::parse_separated_nonempty;
    let input_result = parser.parse(input_tokens).expect("need list of ids");
    let mut strs = vec![]; // This is the vec of op structure name in str.
    for item in input_result {
        match item {
            Expr::Path(expr) => {
                strs.push(expr.path.get_ident().expect("need a ident").to_string());
            },
            _ => {panic!("need a ident, expr::path.");}
        }
    }

    // This is the vec of ident.
    let names: Vec<_> = strs.iter().map(|x| quote::format_ident!("{}", x)).collect();
    
    let serialize_box = quote!{
        pub fn serialize_box<S>(op: &Box<dyn OpTrait>, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer {
            match op.get_name() {
                #( #strs  => {
                    let op = op.as_any().downcast_ref::<#names>().unwrap();
                    op.serialize(serializer)
                },  )*
	        other => {
		    return Err(ser::Error::custom(format!("unknown op {:?}", other)));
	        }
	    }
        }
    };

    let deserialize_map = quote!{
        pub fn deserialize_map<'de, V>(op_name: String, mut map: V) -> Result<Op, V::Error>
        where V: MapAccess<'de>, {
            match op_name.as_str() {
                #( #strs => {
                    let op_obj: #names = Some(map.next_value::<#names>()?).ok_or_else(|| de::Error::missing_field("op_obj"))?;
                    return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
                }, )*
                _ => {
		    return Err(de::Error::missing_field("op_obj"));
		}
            }
        }
    };

    let deserialize_seq = quote!{
	pub fn deserialize_seq<'de, V>(op_name: String, mut seq: V) -> Result<Op, V::Error>
        where V: SeqAccess<'de>, {
            match op_name.as_str() {
                #( #strs => {
                    let op_obj: #names = seq.next_element()?.ok_or_else(|| de::Error::missing_field("op_obj"))?;
                    return Ok(Op::new(Rc::new(RefCell::new(Box::new(op_obj)))));
                }, )*
                _ => {
		    return Err(de::Error::missing_field("op_obj"));
		}
            }
        }
    };

    let tokens = quote! {
        #serialize_box
        #deserialize_map
        #deserialize_seq
    };
    
    tokens.into()
}


#[cfg(test)]
mod tests {
    
    #[test]
    fn test() {
        
    }
}
