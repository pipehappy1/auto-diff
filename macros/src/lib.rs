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
    let mut strs = vec![];
    for item in input_result {
        match item {
            Expr::Path(expr) => {
                strs.push(expr.path.get_ident().expect("need a ident").to_string());
            },
            _ => {panic!("need a ident, expr::path.");}
        }
    }

    let names: Vec<_> = strs.iter().map(|x| quote::format_ident!("{}", x)).collect();
    
    let tokens = quote!{
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

    
    tokens.into()
}


#[cfg(test)]
mod tests {
    
    #[test]
    fn test() {
        
    }
}
