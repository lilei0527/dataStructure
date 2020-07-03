package tree.huffman;

/**
 * @author lilei
 **/
public class Node implements Comparable<Node> {
    public int weight;//权重
    public byte k;//存储实体
    public Node left;
    public Node right;

    public Node(int weight, Node left, Node right) {
        this.weight = weight;
        this.left = left;
        this.right = right;
    }

    public Node(int weight, byte k) {
        this.weight = weight;
        this.k = k;
    }

    public Node(int weight) {
        this.weight = weight;
    }

    @Override
    public int compareTo(Node o) {
        return this.weight - o.weight;
    }

    @Override
    public String toString() {
        return "Node{" +
                "weight=" + weight +
                ", k=" + k +
                ", left=" + left +
                ", right=" + right +
                '}';
    }
}
