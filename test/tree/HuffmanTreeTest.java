package tree;


import tree.huffman.HuffmanTree;

/**
 * @author lilei
 **/
public class HuffmanTreeTest {
    public static void main(String[] args) {
        HuffmanTree<Character> huffmanTree = new HuffmanTree<>('@');
        huffmanTree.add('a', 2);
        huffmanTree.add('b', 5);
        huffmanTree.add('c', 6);
        huffmanTree.add('d', 9);
        huffmanTree.add('e', 10);
        huffmanTree.add('f', 11);
        huffmanTree.add('@', 1);

        huffmanTree.create();

        Character[] characters = {'a','b','c','@'};

        byte[] encode = huffmanTree.encode(characters);
        Object[] decode = huffmanTree.decode(encode);
        for(Object o:decode){
            System.out.println(o.toString());
        }
    }
}
