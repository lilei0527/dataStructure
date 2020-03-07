package tree.akl;

import tree.Entry;
import tree.TreeNode;

@SuppressWarnings("unused")
public class Node<K, V> extends TreeNode<K, V> {
    public int height;

    public Node(Entry<K, V> entry) {
        super(entry);
    }

    @Override
    public String toString() {
        return "Node{" +
                "height=" + height +
                ", entry=" + entry +
                ", right=" + right +
                ", left=" + left +
                '}';
    }
}
