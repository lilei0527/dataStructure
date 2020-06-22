package tree.huffman;

import tree.Coded;
import tree.WeightedTree;

import java.util.*;


/**
 * @author lilei
 **/
public abstract class HuffmanTree<K> implements WeightedTree<K>, Coded<K> {
    public Node<K> root;
    public Queue<Node<K>> queue = new PriorityQueue<>();
    public Map<K, String> codeMap = new HashMap<>();
    public Map<String, K> keyMap = new HashMap<>();
    public String end="1";
    public long fileSize;//文件字节数

    public HuffmanTree(long fileSize) {
        this.fileSize = fileSize;
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
        byte[] bytes = new byte[getCodeLength(ks) / 8 + 1 + end.length()];
        int index = 0;
        for (K k : ks) {
            String code = codeMap.get(k);
            for (int j = 0; j < code.length(); j++) {
                if (code.charAt(j) == '1')
                    bytes[index / 8] |= (1 << (index % 8));
                index++;
            }
        }

        //add end
        for (int j = 0; j < end.length(); j++) {
            if (end.charAt(j) == '1')
                bytes[index / 8] |= (1 << (index % 8));
            index++;
        }
        return bytes;
    }

    public byte[] encode() {
        return encode(toArray());
    }


    @Override
    public byte[] decode(byte[] bytes) {
        int index = 0;
        int i =0;
        byte[] rb = new byte[(int)fileSize];
        StringBuilder stringBuilder = new StringBuilder();
        while (true) {
            if ((bytes[index / 8] & (1 << (index % 8))) == 1 << (index % 8)) {
                stringBuilder.append("1");
            } else {
                stringBuilder.append("0");
            }
            String s = stringBuilder.toString();
            K k = keyMap.get(s);
            if (end.equals(s)) {
                break;
            }
            if (k != null) {
                byte binCode = getBinCode(k);
                rb[i++] = binCode;
                stringBuilder.delete(0, stringBuilder.length());
            }
            index++;
        }
        return rb;
    }

    public Map<K, Integer> count() {
        Map<K, Integer> frequencyMap = new HashMap<>();
        K[] ks = toArray();
        for (K k : ks) {
            frequencyMap.merge(k, 1, Integer::sum);
        }
        return frequencyMap;
    }

    public void addNode() {
        Map<K, Integer> count = count();
        for (Map.Entry<K, Integer> entry : count.entrySet()) {
            add(entry.getKey(), entry.getValue());
        }
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

    public String getCode(K k) {
        return codeMap.get(k);
    }

    public abstract K[] toArray();

    public abstract byte getBinCode(K k);

    @Override
    public String toString() {
        return "HuffmanTree{" +
                "root=" + root +
                ", queue=" + queue +
                ", codeMap=" + codeMap +
                '}';
    }
}
