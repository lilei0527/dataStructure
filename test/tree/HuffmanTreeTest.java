package tree;


import tree.huffman.CharHuffmanTree;
import tree.huffman.HuffmanTree;

/**
 * @author lilei
 **/
public class HuffmanTreeTest {
    public static void main(String[] args) {
        HuffmanTree<Character> tree = new CharHuffmanTree('@',"abc  adfd f @");
        byte[]bytes = tree.encode();
        Object[] decode = tree.decode(bytes);
        System.out.println(tree.getCode('a'));
    }
}
