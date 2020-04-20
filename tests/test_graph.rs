use auto_diff::collection::graph::Graph;
use auto_diff::collection::generational_index::{NetIndex, };



#[test]
fn test_graph() {
    // A   B
    //  \ /
    //   Op
    //   |
    //   C
    {
        let mut g = Graph::new();
        
        let data_A = NetIndex::new(0,0);
        let data_B = NetIndex::new(1,0);
        let data_C = NetIndex::new(2,0);
        g.add_data(&data_A);
        g.add_data(&data_B);
        g.add_data(&data_C);
        
        let op_A = NetIndex::new(0,0);
        g.add_op(&op_A);

        g.connect(&[&data_A, &data_B], &[&data_C,], &op_A);

        g.walk(
            &[&data_A, &data_B],
            true,
            |input, output, op| {
                println!("forward: {:?}, {:?}, {}", input, output, op);
                assert_eq!(input.len(), 2);
                assert_eq!(input[0], NetIndex::new(0,0));
                assert_eq!(input[1], NetIndex::new(1,0));
                assert_eq!(output[0], NetIndex::new(2,0));
                assert_eq!(*op, NetIndex::new(0,0));
            }
        );

        g.walk(
            &[&data_C],
            false,
            |input, output, op| {
                println!("backward: {:?}, {:?}, {}", input, output, op);
                assert_eq!(input.len(), 1);
                assert_eq!(input[0], NetIndex::new(2,0));
                assert_eq!(output[0], NetIndex::new(0,0));
                assert_eq!(output[1], NetIndex::new(1,0));
                assert_eq!(*op, NetIndex::new(0,0));
            }
        );
    }

    // A   B
    //  \ /
    //   Op1
    //   |
    //   C   D
    //    \ /
    //     Op2
    //     |
    //     E
    {
        
    }
}
