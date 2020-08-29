package file;


/**
 * @author lilei
 * 用于文件搜索的条件设置
 * 文件属性：文件名，文件类型，文件修改时间，文件大小，文件父路径属性
 * 文件名支持模糊搜索，全值搜索
 * 文件类型名支持模糊搜索，全值搜索
 * 文件修改时间支持按照时间范围查找
 * 文件大小支持按照文件大小范围查找
 **/
public abstract class Criteria {


    private FileAttribute attribute;
    private String value;

    public Criteria(FileAttribute attribute, String value) {
        this.attribute = attribute;
        this.value = value;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }
}
