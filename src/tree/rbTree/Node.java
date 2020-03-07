package tree.rbTree;

import tree.Entry;
import tree.TreeNode;

@SuppressWarnings("unused")
public class Node<K,V> extends TreeNode<K,V> {
    public boolean color; //true-黑  false-红

    @Override
    public String toString() {
        return "Node{" +
                ", left=" + left +
                ", right=" + right +
                ", entry=" + entry +
                ", color=" + color +
                '}';
    }


    public Node(Node<K,V> parent, Entry<K,V> entry) {
        super(entry);
        this.parent = parent;
        color = true;
    }
}
