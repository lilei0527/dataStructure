package tree.akl;

import tree.Entry;

@SuppressWarnings("unused")
public class Node {

    private int height;
    private Entry entry;
    private Node right;
    private Node left;
    private Node parent;

    public Node(Entry entry) {
        this.entry = entry;
    }

    public Entry getEntry() {
        return entry;
    }

    public void setEntry(Entry entry) {
        this.entry = entry;
    }

    public int getHeight() {
        return height;
    }

    public void setHeight(int height) {
        this.height = height;
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

    public Node getParent() {
        return parent;
    }

    public void setParent(Node parent) {
        this.parent = parent;
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
