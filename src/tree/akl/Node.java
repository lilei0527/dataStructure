package tree.akl;

import tree.Entry;

@SuppressWarnings("unused")
public class Node {

    public int height;
    public Entry entry;
    public Node right;
    public Node left;

    public Node(Entry entry) {
        this.entry = entry;
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
