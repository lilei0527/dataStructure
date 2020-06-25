package tree.huffman;

import java.io.*;

/**
 * @author lilei
 **/
@SuppressWarnings("unused")
public class ByteHuffmanTree extends HuffmanTree<Byte> {
    private File file;


    public ByteHuffmanTree(File file) {
        super(file.length());
        this.file = file;
        addNode();
        create();
    }

    @Override
    public byte[] toArray() {
        InputStream inputStream = null;
        try {
            inputStream = new FileInputStream(file);
            byte[] bytes = new byte[(int) file.length()];
            inputStream.read(bytes);
            return bytes;
        } catch (IOException e) {
            return null;
        } finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
