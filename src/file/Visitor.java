package file;

import java.io.File;

public interface Visitor {
    boolean visitName(File file);
    boolean visitType(File file);
    boolean visitLength(File file);
    boolean visitUpdateTime(File file);
}
