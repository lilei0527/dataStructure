package tree.huffman;

import tree.Coded;
import tree.WeightedTree;

import java.util.*;


/**
 * @author lilei
 **/
public abstract class HuffmanTree<K> implements WeightedTree<K>, Coded<K> {
    public Node<Byte> root;
    public Queue<Node<Byte>> queue = new PriorityQueue<>();
    public Map<Byte, String> codeMap = new HashMap<>();
    public Map<String, Byte> keyMap = new HashMap<>();
    public long fileSize;//文件字节数

    public HuffmanTree(long fileSize) {
        this.fileSize = fileSize;
    }

    @Override
    public void create() {
        while (queue.peek() != null) {
            Node<Byte> left = queue.poll();
            Node<Byte> right = queue.poll();
            if (right == null) {
                root = left;
                break;
            }
            Node<Byte> parent = new Node<>(left.weight + right.weight, left, right);
            queue.add(parent);
        }
        saveCode();
    }

    public void saveCode() {
        String code = "";
        saveCode(root, code);
    }

    @Override
    public byte[] encode(byte[] ks) {
//        int codeLength = getCodeLength(ks);
//        int size = codeLength % 8 == 0 ? codeLength / 8 : codeLength / 8 + 1;

        int decreaseByte = getDecreaseByte(ks);
        int size = ks.length-decreaseByte;
        byte[] bytes = new byte[size];
        int index = 0;
        int i = 0;
//        out:
        for (Byte k : ks) {
            String code = codeMap.get(k);
            for (int j = 0; j < code.length(); j++) {
                if (code.charAt(j) == '1'){
                    bytes[index / 8] |= (1 << (index % 8));
                    i++;
                }
                if(i>=size) break;
                index++;
//                if (s++<=size) {
//                    break out;
//                }
            }
        }

        return bytes;
    }

    public byte[] encode() {
        return encode(toArray());
    }


    @Override
    public byte[] decode(byte[] bytes) {
        int index = 0;
        int i = 0;
        byte[] rb = new byte[(int) fileSize];
        StringBuilder stringBuilder = new StringBuilder();
        do {
            if ((bytes[index / 8] & (1 << (index % 8))) == 1 << (index % 8)) {
                stringBuilder.append("1");
            } else {
                stringBuilder.append("0");
            }
            index++;
            String s = stringBuilder.toString();
            Byte k = keyMap.get(s);

            if (k != null) {
                rb[i++] = k;
                stringBuilder.delete(0, stringBuilder.length());
            }
        } while (i < fileSize);
        return rb;
    }

    public Map<Byte, Integer> count() {
        Map<Byte, Integer> frequencyMap = new HashMap<>();
        byte[] ks = toArray();
        for (byte k : ks) {
            frequencyMap.merge(k, 1, Integer::sum);
        }
        return frequencyMap;
    }

    public void addNode() {
        Map<Byte, Integer> count = count();
        for (Map.Entry<Byte, Integer> entry : count.entrySet()) {
            add(entry.getKey(), entry.getValue());
        }
    }

    private int getCodeLength(byte[] ks) {
        int length = 0;
        for (byte k : ks) {
            length += codeMap.get(k).length();
        }
        return length;
    }

    private int getDecreaseByte(byte[] ks) {
        int length = 0;
        for (byte k : ks) {
            length += (8 - codeMap.get(k).length());
        }
        return length % 8 == 0 ? length / 8 : length / 8 + 1;
    }

    public void saveCode(Node<Byte> node, String code) {
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


    public void add(Byte k, int weight) {
        queue.add(new Node<>(weight, k));
    }


    public abstract byte[] toArray();

    @Override
    public String toString() {
        return "HuffmanTree{" +
                "root=" + root +
                ", queue=" + queue +
                ", codeMap=" + codeMap +
                '}';
    }
}
