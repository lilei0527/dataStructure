package file;

import java.io.File;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class NotLikeCriteria extends ComparableCriteria implements Visitor{
    public NotLikeCriteria(FileAttribute attribute, Object object) {
        super(attribute, object);
    }

    @Override
    public boolean visitName(File file) {
        return isNotMather(file.getName(), (String) object);
    }

    @Override
    public boolean visitType(File file) {
        return isNotMather(getSuffix(file), (String) object);
    }

    @Override
    public boolean visitLength(File file) {
        throw new IllegalArgumentException("fileLength can not regex match");
    }

    @Override
    public boolean visitUpdateTime(File file) {
        throw new IllegalArgumentException("updateTime can not regex match");
    }

    private boolean isNotMather(String s, String pattern) {
        Pattern p = Pattern.compile(pattern);
        Matcher matcher = p.matcher(s);
        return !matcher.find();
    }
}
