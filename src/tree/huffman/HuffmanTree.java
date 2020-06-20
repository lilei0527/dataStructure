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
    public Map<K, byte[]> codeMap = new HashMap<>();

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
        byte[] bytes = new byte[1];
        bytes[0] = 0x01;
        saveCode(root, bytes);
    }

    @Override
    public byte[] code(K[] key) {
        byte[]
    }

    private byte[] mergeBytes(byte[] original, byte[] merged) {
        int originalZero = getZero(original[original.length - 1]);
        int mergedZero = getZero(original[original.length - 1]);
        int mergeSize = (8 - originalZero) + (8 - mergedZero);
        int moreSize = mergeSize > 8 ? 2 : 1;
        int size = original.length-1+merged.length-1+moreSize;
        byte[]bytes = new byte[size];
        System.arraycopy(original,0,bytes,0,original.length);
        for(int i=merged.length-1;i>0;i--){
            for(int j=originalZero;j<8;j++){
                
            }
        }
    }

    public int getZero(byte b) {
        int i = 7;
        int sum = 0;

        while ((b & 0xff & (1 << i)) == 0) {
            sum++;
            i--;
        }
        return sum;
    }

    @Override
    public byte[] getCode(K k) {
        return codeMap.get(k);
    }


    public void saveCode(Node<K> node, byte[] code) {
        if (node.left == null && node.right == null) {
            codeMap.put(node.k, code);
        }
        if (node.left != null) {
            byte[] newCode = code(code, (byte) 0x00);
            saveCode(node.left, newCode);
        }
        if (node.right != null) {
            byte[] newCode = code(code, (byte) 0x01);
            saveCode(node.right, newCode);
        }
    }

    private byte[] code(byte[] code, byte bit) {
        byte[] bytes = moveLeftOneBit(code);
        bytes[0] |= bit;
        return bytes;
    }

    private byte[] moveLeftOneBit(byte[] code) {
        if (getZero(code[code.length - 1]) == 0) {
            byte[] newBytes = new byte[code.length + 1];
            System.arraycopy(code, 0, newBytes, 0, code.length);
            code = newBytes;
        }
        for (int i = code.length - 1; i >= 0; i--) {
            code[i] = (byte) (code[i] << 1);
            if (i > 0)
                code[i] |= ((code[i - 1] & 0xff) >> 7);
        }
        return code;
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

    public static void main(String[] args) {
        HuffmanTree<Character> huffmanTree = new HuffmanTree<>();
        byte[] bytes = new byte[1];
        bytes[0] = (byte) 0b10101010;
        huffmanTree.moveLeftOneBit(bytes);
    }
}
