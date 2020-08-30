package file;


import java.io.File;
import java.util.List;

/**
 * @author lilei
 * 用于文件搜索的条件设置
 * 文件属性：文件名，文件类型，文件修改时间，文件大小，文件父路径属性
 * 文件名支持模糊搜索，全值搜索
 * 文件类型名支持模糊搜索，全值搜索
 * 文件修改时间支持按照时间范围查找
 * 文件大小支持按照文件大小范围查找
 **/
public interface Criteria {

    enum BooleanOperator {
        AND, OR,NONE
    }

    BooleanOperator getOperator();

    List<Criteria> getExpressions();

    boolean isCompound();

    boolean isMatch(File file);
}
