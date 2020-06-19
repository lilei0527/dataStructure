package tree;


import tree.huffman.HuffmanTree;

/**
 * @author lilei
 **/
public class HuffmanTreeTest {
    public static void main(String[] args) {
        HuffmanTree<Character> huffmanTree = new HuffmanTree<>();
        huffmanTree.add('a', 2);
        huffmanTree.add('b', 5);
        huffmanTree.add('c', 6);
        huffmanTree.add('d', 9);
        huffmanTree.add('e', 10);
        huffmanTree.add('f', 11);

        huffmanTree.create();
    }
}
