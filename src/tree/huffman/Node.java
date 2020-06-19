package tree.huffman;

/**
 * @author lilei
 **/
public class Node<K> implements Comparable<Node<K>>{
    public int weight;//权重
    public K k;//存储实体
    public Node<K> left;
    public Node<K> right;

    public Node(int weight, K k, Node<K> left, Node<K> right) {
        this.weight = weight;
        this.k = k;
        this.left = left;
        this.right = right;
    }

    public Node(Node<K> left, Node<K> right) {
        this.left = left;
        this.right = right;
    }

    public Node(int weight, Node<K> left, Node<K> right) {
        this.weight = weight;
        this.left = left;
        this.right = right;
    }

    public Node(int weight, K k) {
        this.weight = weight;
        this.k = k;
    }

    @Override
    public int compareTo(Node<K> o) {
        return this.weight-o.weight;
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
