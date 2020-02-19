package tree.akl;

import tree.Entry;

@SuppressWarnings("unused")
public class Node {


    private Entry entry;
    private Node right;
    private Node left;

    public Node(Entry entry, Node right, Node left) {
        this.entry = entry;
        this.right = right;
        this.left = left;
    }

    public Entry getEntry() {
        return entry;
    }

    public void setEntry(Entry entry) {
        this.entry = entry;
    }


    public Node getRight() {
        return right;
    }

    public void setRight(Node right) {
        this.right = right;
    }

    public Node getLeft() {
        return left;
    }

    public void setLeft(Node left) {
        this.left = left;
    }

    @Override
    public String toString() {
        return "Node{" +
                "entry=" + entry +
                ", right=" + right +
                ", left=" + left +
                '}';
    }
}
