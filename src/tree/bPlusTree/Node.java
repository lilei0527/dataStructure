package tree.bPlusTree;

import tree.Entry;

import java.util.List;

/*节点类*/
public class Node<K , V> {

    //节点的子节点
    private List<Node<K, V>> nodes;
    //节点的键值对
    private List<Entry<K, V>> entry;
    //节点的后节点
    private Node<K, V> nextNode;
    //节点的前节点
    private Node<K, V> previousNode;
    //节点的父节点
    private Node<K, V> parentNode;

    public Node(List<Node<K, V>> nodes, List<Entry<K, V>> entry, Node<K, V> nextNode, Node<K, V> previousNode, Node<K, V> parentNode) {
        this.nodes = nodes;
        this.entry = entry;
        this.nextNode = nextNode;
        this.parentNode = parentNode;
        this.previousNode = previousNode;
    }

     boolean isLeaf() {
        return nodes==null;
    }

     boolean isHead() {
        return previousNode == null;
    }

     boolean isTail() {
        return nextNode == null;
    }

     boolean isRoot() {
        return parentNode == null;
    }


     List<Node<K, V>> getNodes() {
        return nodes;
    }

     void setNodes(List<Node<K, V>> nodes) {
        this.nodes = nodes;
    }


    List<Entry<K, V>> getEntry() {
        return entry;
    }

//    public void setKeyAndValue(List<KeyAndValue> KeyAndValue) {
//        this.keyAndValue = KeyAndValue;
//    }

    Node<K, V> getNextNode() {
        return nextNode;
    }

     void setNextNode(Node<K, V> nextNode) {
        this.nextNode = nextNode;
    }

     Node<K, V> getParentNode() {
        return parentNode;
    }

     void setParentNode(Node<K, V> parentNode) {
        this.parentNode = parentNode;
    }

     Node<K, V> getPreviousNode() {
        return previousNode;
    }

     void setPreviousNode(Node<K, V> previousNode) {
        this.previousNode = previousNode;
    }
}
