package tree;


import tree.huffman.ByteHuffmanTree;
import tree.huffman.CharHuffmanTree;
import tree.huffman.HuffmanTree;

import java.io.*;

/**
 * @author lilei
 **/
public class HuffmanTreeTest {
    public static void main(String[] args) throws IOException {
//        HuffmanTree<Character> tree = new CharHuffmanTree("abc  adfd f @");
//        byte[] bytes = tree.encode();
//        byte[] decode = tree.decode(bytes);
//        for(byte o:decode){
//            System.out.println((char) o);
//        }

        HuffmanTree<Byte> tree = new ByteHuffmanTree(new File("C:\\Users\\lilei\\Desktop\\1.jpg"));
        byte[] bytes = tree.encode();

        File file1 = new File("C:\\Users\\lilei\\Desktop\\1.zip");
        OutputStream outputStream1 = new FileOutputStream(file1);
        outputStream1.write(bytes);

        byte[] decode = tree.decode(bytes);
        File file = new File("C:\\Users\\lilei\\Desktop\\2.jpg");
        OutputStream outputStream = new FileOutputStream(file);
        outputStream.write(decode);
    }
}
