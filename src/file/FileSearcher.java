package file;

import java.io.File;
import java.util.*;

/**
 * @author lilei
 **/
public class FileSearcher {

    private static List<File> find(String path, Criteria criteria) {
        List<File> list = new ArrayList<>();
        find(path, criteria, list);
        return list;
    }

    private static void find(String path, Criteria criteria, List<File> files) {
        File file = new File(path);
        if (file.isDirectory()) {
            for (File f : Objects.requireNonNull(file.listFiles())) {
                find(path + File.separatorChar + f.getName(), criteria, files);
            }
        } else {
            if (criteria.isMatch(file)) {
                files.add(file);
            }
        }
    }


    public static void main(String[] args) {

        CriteriaBuilder criteriaBuilder = new CriteriaBuilder();
        Criteria criteria = criteriaBuilder.like(FileAttribute.NAME, ".*liu.*");
        Criteria criteria1 = criteriaBuilder.eq(FileAttribute.NAME, "Desktop");
        Criteria criteria2 = criteriaBuilder.parent(criteria1, 1);
        Criteria or = criteriaBuilder.or(criteria, criteria2);

        List<File> files = find("C:\\Users\\lilei\\Desktop", or);
        System.out.println(files);

    }
}
