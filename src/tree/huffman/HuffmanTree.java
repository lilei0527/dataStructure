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
    public Map<String, K> keyMap = new HashMap<>();
    public K end;

    public HuffmanTree(K end) {
        this.end = end;
    }

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
                if (poll == null) {
                    root = node;
                    break;
                }
                l = poll;
                r = node;
            }
        }

        saveCode();
    }

    public void saveCode() {
        String code = "0";
        saveCode(root, code);
    }

    @Override
    public byte[] encode(K[] ks) {
        byte[] bytes = new byte[getCodeLength(ks) / 8 + 1];
        int index = 0;
        for (K k : ks) {
            String code = codeMap.get(k);
            for (int j = 0; j < code.length(); j++) {
                if (code.charAt(j) == '1')
                    bytes[index / 8] |= (1 << (index % 8));
                index++;
            }
        }
        return bytes;
    }

    @Override
    public Object[] decode(byte[] bytes) {
        int index = 0;
        List<K> list = new ArrayList<>();
        StringBuilder stringBuilder = new StringBuilder();
        while (true) {
            if ((bytes[index / 8] & (1 << (index % 8))) == 1 << (index % 8)) {
                stringBuilder.append("1");
            } else {
                stringBuilder.append("0");
            }
            index++;
            String s = stringBuilder.toString();
            K k = keyMap.get(s);
            if (end.equals(k)) {
                break;
            }
            if (k != null) {
                list.add(k);
                stringBuilder.delete(0, stringBuilder.length());
            }
        }
        return list.toArray();
    }

    private int getCodeLength(K[] ks) {
        int length = 0;
        for (K k : ks) {
            length += codeMap.get(k).length();
        }
        return length;
    }

    public void saveCode(Node<K> node, String code) {
        if (node.left == null && node.right == null) {
            codeMap.put(node.k, code);
            keyMap.put(code, node.k);
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
        Node<K> parent = new Node<>(node.weight + newNode.weight, node, newNode);
        Node<K> l = queue.poll();
        Node<K> r = queue.poll();
        if (l == null) {
            return parent;
        } else if (r == null) {
            return new Node<>(l.weight + parent.weight, l, parent);
        } else {
            Node<K> mid = new Node<>(l.weight + r.weight, l, r);
            return new Node<>(mid.weight + parent.weight, mid, parent);
        }
    }

    public void add(K k, int weight) {
        queue.add(new Node<>(weight, k));
    }


    @Override
    public String toString() {
        return "HuffmanTree{" +
                "root=" + root +
                ", queue=" + queue +
                ", codeMap=" + codeMap +
                '}';
    }
}
