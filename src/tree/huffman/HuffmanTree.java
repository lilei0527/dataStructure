package tree.huffman;

import tree.Coded;
import tree.WeightedTree;

import java.util.*;


/**
 * @author lilei
 **/
public class HuffmanTree<K> implements WeightedTree<K>, Coded<K> {
    public Node<K> root;
    public Queue<Node<K>> queue = new PriorityQueue<>();
    public Map<K, String> codeMap = new HashMap<>();

    @Override
    public void create() {
        Node<K> l = queue.poll();
        Node<K> r = queue.poll();

        if (l == null) {
            root = null;
        } else if (r == null) {
            root = l;
        } else {

            while (true) {
                Node<K> node = create(l, r);
                Node<K> poll = queue.poll();
                if (node.weight == l.weight + r.weight || poll == null) {
                    root = node;
                    break;
                }
                if (node.weight > poll.weight) {
                    l = poll;
                    r = node;
                } else {
                    l = node;
                    r = poll;
                }
            }
        }

        listCode();
    }

    @Override
    public void listCode() {
        String code = "0";
        saveCode(root, code);
    }

    public void saveCode(Node<K> node, String code) {
        if (node.left == null && node.right == null) {
            codeMap.put(node.k, code);
        }
        if (node.left != null) {
            saveCode(node.left, code + "0");
        }
        if (node.right != null) {
            saveCode(node.right, code + "1");
        }
    }


    public Node<K> create(Node<K> left, Node<K> right) {
        Node<K> node = queue.poll();

        Node<K> newNode = new Node<>(left.weight + right.weight, left, right);

        if (node == null) {
            return newNode;
        }
        Node<K> parent;
        if ((left.weight + right.weight) > node.weight) {
            parent = new Node<>(node.weight + newNode.weight, node, newNode);
        } else {
            parent = new Node<>(node.weight + newNode.weight, newNode, node);
        }
        return parent;
    }

    public void add(K k, int weight) {
        queue.add(new Node<>(weight, k));
    }


    @Override
    public String toString() {
        return "HuffmanTree{" +
                "root=" + root +
                '}';
    }
}
