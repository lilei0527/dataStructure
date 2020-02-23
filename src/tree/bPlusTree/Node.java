package tree.bPlusTree;

import tree.Entry;

import java.util.List;

/*节点类*/
public class Node {

    //节点的子节点
    private List<Node> nodes;
    //节点的键值对
    private List<Entry> entry;
    //节点的后节点
    private Node nextNode;
    //节点的前节点
    private Node previousNode;
    //节点的父节点
    private Node parentNode;

    public Node(List<Node> nodes, List<Entry> entry, Node nextNode, Node previousNode, Node parentNode) {
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


     List<Node> getNodes() {
        return nodes;
    }

     void setNodes(List<Node> nodes) {
        this.nodes = nodes;
    }


    List<Entry> getEntry() {
        return entry;
    }

//    public void setKeyAndValue(List<KeyAndValue> KeyAndValue) {
//        this.keyAndValue = KeyAndValue;
//    }

    Node getNextNode() {
        return nextNode;
    }

     void setNextNode(Node nextNode) {
        this.nextNode = nextNode;
    }

     Node getParentNode() {
        return parentNode;
    }

     void setParentNode(Node parentNode) {
        this.parentNode = parentNode;
    }

     Node getPreviousNode() {
        return previousNode;
    }

     void setPreviousNode(Node previousNode) {
        this.previousNode = previousNode;
    }
}
