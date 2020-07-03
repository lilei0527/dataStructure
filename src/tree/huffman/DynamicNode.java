package tree.huffman;

/**
 * @author lilei
 **/
public class DynamicNode extends Node{
    public int number;//节点
    public DynamicNode parent;//父节点

    public DynamicNode(int weight, int number, DynamicNode parent) {
        super(weight);
        this.number = number;
        this.parent = parent;
    }

    public DynamicNode(int weight, byte k, int number) {
        super(weight, k);
        this.number = number;
    }

    public DynamicNode(int weight, Node left, Node right, int number, DynamicNode parent) {
        super(weight, left, right);
        this.number = number;
        this.parent = parent;
    }

    public DynamicNode(int weight, byte k, int number, DynamicNode parent) {
        super(weight, k);
        this.number = number;
        this.parent = parent;
    }

}
