package file;

import java.io.File;
import java.util.*;

/**
 * @author lilei
 **/
public class FileSearcher {


    private static List<File> find(String path, Query query) {
        List<File> list = new ArrayList<>();

    }

    private static List<File> find(String path, Query query, List<File> files) {
        File file = new File(path);
        if (file.isDirectory()) {
            for (File f : Objects.requireNonNull(file.listFiles())) {
                find(path + File.separatorChar + f.getName(), query, files);
            }
        } else {

        }
    }

    private static boolean isFind(File file, Query query) {

        List<Criteria> criteriaList = query.getCriteriaList();
        List<Query.BooleanOperator> operator = query.getOperator();

        for (Criteria criteria : criteriaList) {

            Map<String, String> criteriaMap = criteria.getCriteriaMap();
            Collection<String> keySet = criteriaMap.keySet();

            for (String key : keySet) {
                switch (key) {
                    case "$eq": {
                        if (criteria.getAttribute().equals(FileAttribute.NAME)) {
                            if (!file.getName().equals(criteriaMap.get(key))) {
                                return false;
                            }
                        }

                        if (criteria.getAttribute().equals(FileAttribute.TYPE)) {
                            if (!getSuffix(file).equals(criteriaMap.get(key))) {
                                return false;
                            }
                        }

                        if (criteria.getAttribute().equals(FileAttribute.SIZE)) {
                            if (String.valueOf(file.length()).equals(criteriaMap.get(key))) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
    }


    private static String getSuffix(File file) {
        String name = file.getName();
        return name.substring(name.lastIndexOf('.'));
    }

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
