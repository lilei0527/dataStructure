package tree.rbTree;

import tree.Entry;
@SuppressWarnings("unused")
public class Node {
    public Node parent;
    public Node left;
    public Node right;
    public Entry entry;
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


    public Node(Node parent, Entry entry) {
        this.parent = parent;
        this.entry = entry;
        color = true;
    }
}
