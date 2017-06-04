package net.librec.recommender.cf.rating;

import net.librec.common.LibrecException;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.MatrixEntry;
import net.librec.recommender.MatrixFactorizationRecommender;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by Administrator on 2017/6/3 0003.
 */
public class COFACTORRecommender extends MatrixFactorizationRecommender {
    public DenseMatrix SPPMImatrix;
    public int k=5;// the number of negative samples
    public DenseMatrix w; //bias value of item
    public DenseMatrix c;  //bias value of context
    public DenseMatrix g; // context embedding
    public float regG;
    public List<Integer> ulist1 =new ArrayList<>();
    public List<Integer> ulist2 =new ArrayList<>();
    public int len=0;
    public int val=0;
    public float a;
    public float b;
    public double maxd=0;
    public int l=1;



    @Override
    public void setup() throws LibrecException {
        // 初始化各种参数 因为extends MatrixFactorizationRecommender 所以初始化训练矩阵 测试矩阵 用户特征矩阵 项目特征矩阵
        super.setup();
        w= new DenseMatrix(numItems,1);
        c=new DenseMatrix(numItems,1);
        g=new DenseMatrix(numItems,numFactors);
        regG=0.03f;
        w.init(initMean, initStd);
        c.init(initMean, initStd);
        g.init(initMean,initStd);
        SPPMImatrix = new DenseMatrix(numItems, numItems);
        int dj=0;
        for (MatrixEntry me : trainMatrix) {
            int itemId1 = me.column();
            System.out.println(itemId1+"   times: "+(dj++));

            ulist1=trainMatrix.getRows(itemId1);//所有购买item的用户
            if(ulist1.size()<10)
            {
                continue;
            }


            for (MatrixEntry me2:trainMatrix)
            {
                int itemId2=me2.column();
                if(itemId2==itemId1)
                {
                    continue;
                }
                ulist2=trainMatrix.getRows(itemId2);
                len=countsame(ulist1,ulist2);
                if(len>0)
                {
                    float dnumItems=numItems;
                    float dlen=len;
                    float us1=ulist1.size();
                    float us2=ulist2.size();
                    a=((dnumItems*dlen)/(us1*us2));
                    double d=Math.log(a)/Math.log(2)-Math.log(5)/Math.log(2);
                    BigDecimal b   =   new   BigDecimal(d);
                    double   d1   =   b.setScale(2,   BigDecimal.ROUND_HALF_UP).doubleValue();
                    if(d1>0)
                    {
                        d1=d1;
                    }
                    else {
                        d1=0;
                    }
                    if(d1>0)
                    {
                        if (maxd<d1)
                        {
                            maxd=d1;
                        }
                        SPPMImatrix.add1(itemId1,itemId2,d1);
                        SPPMImatrix.add1(itemId2,itemId1,d1);

                    }

                }
            }

        }

        for(int i=0;i<numItems;i++)
        {
            if(i%100==0)
            {
                System.out.println("在归一化");
            }
            for (int j=0;j<numItems;j++)
            {
                double p=SPPMImatrix.get(i,j);
                double newp=p/maxd;
                BigDecimal b   =   new   BigDecimal(newp);
                double   newp1   =   b.setScale(2,   BigDecimal.ROUND_HALF_UP).doubleValue();
                SPPMImatrix.add1(i,j,newp1);
            }
        }
    }



    public int countsame(List<Integer> l1,List<Integer> l2)
    {
        Set<Integer> same = new HashSet<Integer>();  //用来存放两个数组中相同的元素
        Set<Integer> temp = new HashSet<Integer>();  //用来存放数组a中的元素

        for (int i = 0; i < l1.size(); i++) {
            temp.add(l1.get(i));   //把数组a中的元素放到Set中，可以去除重复的元素
        }

        for (int j = 0; j < l2.size(); j++) {
            //把数组b中的元素添加到temp中
            //如果temp中已存在相同的元素，则temp.add（b[j]）返回false
            if(!temp.add(l2.get(j)))
                same.add(l2.get(j));
        }
        return same.size();
    }


    @Override
    protected void trainModel() throws LibrecException {
        for (int iter = 1; iter <= numIterations; iter++) {

            loss = 0.0d;
            for (MatrixEntry me : trainMatrix) {
                int userId = me.row(); // user
                int itemId = me.column(); // item
                double realRating = me.get();

                double predictRating = predict(userId, itemId); //利用用户特征向量乘以项目特征向量
                double error = realRating - predictRating;
                loss += error * error;

                // update factors  梯度下降
                for (int factorId = 0; factorId < numFactors; factorId++) {
                    double userFactor = userFactors.get(userId, factorId), itemFactor = itemFactors.get(itemId, factorId);

                    userFactors.add(userId, factorId, learnRate * (error * itemFactor - regUser * userFactor));
                    itemFactors.add(itemId, factorId, learnRate * (error * userFactor - regItem * itemFactor));
                    loss += regUser * userFactor * userFactor + regItem * itemFactor * itemFactor;
                }
            }


            for(int i=0;i<numItems;i++)
            {
                int tou=i;

                for(int j=0;j<numItems;j++)
                {
                    double mij=SPPMImatrix.get(i,j);
                    if(mij==0)
                    {
                        continue;
                    }
                    double predictrating2=predict1(i,j);
                    double wi=w.get(i,0);
                    double cj=c.get(j,0);
                    double diff=mij-itemFactors.get(j,k)*g.get(j,k)-wi-cj;
//                    loss+=diff*diff+0.03*predict2(i,j);
                    loss+=diff*diff;
                    w.add(i,0,learnRate*diff);
                    c.add(i,0,learnRate*diff);
                    for(int k=0;k<numFactors;k++)
                    {
                        itemFactors.add(j,k,learnRate*diff*g.get(j,k));
                        g.add(j,k,learnRate*diff*itemFactors.get(j,k));
                        loss+=0.03*g.get(j,k)*g.get(j,k);

                    }
                }
            }


            loss *= 0.5;
            if (isConverged(iter) && earlyStop) {
                break;    // deltaloss 小于1e-5 认为收敛
            }
            updateLRate(iter);
        }

    }

}
