package tree.huffman;

import java.util.HashMap;
import java.util.Map;

/**
 * @author lilei
 **/
public class CharHuffmanTree extends HuffmanTree<Character> {
    private String content;

    public CharHuffmanTree(Character end, String content) {
        super(end);
        this.content = content;
        addNode();
        create();
    }

    public Character[] toChar() {
        char[] chars = content.toCharArray();
        Character[] characters = new Character[chars.length];
        int index = 0;
        for (char c : chars) {
            characters[index++] = c;
        }
        return characters;
    }

    public Map<Character, Integer> count() {
        Map<Character, Integer> frequencyMap = new HashMap<>();
        Character[] characters = toChar();
        for (Character character : characters) {
            frequencyMap.merge(character, 1, Integer::sum);
        }
        return frequencyMap;
    }


    public void addNode(){
        Map<Character, Integer> count = count();
        for(Map.Entry<Character, Integer> entry:count.entrySet()){
            add(entry.getKey(),entry.getValue());
        }
    }

    public byte[] encode(){
        return encode(toChar());
    }

}
