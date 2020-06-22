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
    public Byte[] toArray() {
        InputStream inputStream = null;
        try {
            inputStream = new FileInputStream(file);
            byte[] bytes = new byte[(int) file.length()];
            inputStream.read(bytes);
            Byte[] rb = new Byte[bytes.length];
            int index = 0;
            for (byte b : bytes) {
                rb[index++] = b;
            }
            return rb;
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

    @Override
    public byte getBinCode(Byte aByte) {
        return aByte;
    }

}
