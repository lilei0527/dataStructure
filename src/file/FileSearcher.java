package file;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * @author lilei
 **/
public class FileSearcher {


    /**
     * @author lilei
     * create at 17:32 2020/8/27
     * 查询path路径下，父文件夹名为name的文件
     **/
    private static List<File> byDirName(String path, String name) {
        List<File> list = new ArrayList<>();
        byDirName(path, name, list);
        return list;
    }

    private static void byDirName(String path, String name, List<File> files) {
        File file = new File(path);
        if (file.isDirectory()) {
            for (File f : Objects.requireNonNull(file.listFiles())) {
                byDirName(path + File.separatorChar + f.getName(), name, files);
            }
        } else {
            if (file.getParentFile().getName().equals(name)) {
                files.add(file);
            }
        }
    }

    public static void main(String[] args) {
        List<File> list = byDirName("C:\\Users\\lilei\\Desktop\\project\\version", "config");
        System.out.println(list);
    }
}
