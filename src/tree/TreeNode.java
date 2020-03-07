package tree;

import tree.rbTree.Node;

public class TreeNode<K, V> {
    public Node<K,V> parent;
    public Entry<K, V> entry;
    public TreeNode<K, V> right;
    public TreeNode<K, V> left;

    public TreeNode(Entry<K,V> entry) {
        this.entry = entry;
    }
}
