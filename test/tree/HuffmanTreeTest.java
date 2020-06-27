package tree;


import tree.huffman.HuffmanTree;

import java.io.*;

/**
 * @author lilei
 **/
public class HuffmanTreeTest {
    public static void main(String[] args) throws IOException {
        HuffmanTree tree = new HuffmanTree();
        byte[] bytes = tree.encode("a adfsd afdf ");
        byte[] decode = tree.decode(bytes);
        for(byte o:decode){
            System.out.println((char) o);
        }

        byte[] bytes1 = tree.encode(new File("C:\\Users\\lilei\\Downloads\\基于改进Canny算子的图像边缘检测算法.pdf"));

        File file1 = new File("C:\\Users\\lilei\\Desktop\\1.zip");
        OutputStream outputStream1 = new FileOutputStream(file1);
        outputStream1.write(bytes1);

        byte[] decode1 = tree.decode(bytes1);
        File file = new File("C:\\Users\\lilei\\Desktop\\2.pdf");
        OutputStream outputStream = new FileOutputStream(file);
        outputStream.write(decode1);
    }
}
