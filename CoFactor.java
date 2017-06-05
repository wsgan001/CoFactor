package net.librec.recommender.cf.rating;

import net.librec.common.LibrecException;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.DenseVector;
import net.librec.math.structure.MatrixEntry;
import net.librec.recommender.MatrixFactorizationRecommender;

import java.io.*;
import java.math.BigDecimal;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by murdonson on 2017/6/3 0003.
 */
public class COFACTORRecommender extends MatrixFactorizationRecommender {
    public DenseMatrix SPPMImatrix;
    public int k=5;// the number of negative samples
    public DenseVector w; //bias value of item
    public DenseVector c;  //bias value of context
    //public DenseMatrix g; // context embedding 我在 MatrixFactorizationReco...里面定义了g
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
        super.setup();
        //w = new DenseMatrix(numItems, 1);
        w= new DenseVector(numItems);
        //c = new DenseMatrix(numItems, 1);
        c = new DenseVector(numItems);
        g = new DenseMatrix(numItems, numFactors);
        regG = 0.03f;
        w.init(initMean, initStd);
        c.init(initMean, initStd);
        g.init(initMean, initStd);
        SPPMImatrix = new DenseMatrix(numItems, numItems);
        //构建SPPMI矩阵  A1   最开始注释A3 生成txt文件  然后注释A1 A2 读取txt文件 节省时间

        int dj = 0;
        int murdonson=0;
        for (MatrixEntry me : trainMatrix) {
            int itemId1 = me.column();

            if(itemId1%150==0)
            {
                System.out.println("程序在跑   "+(murdonson++)); //看程序速度如何
            }
            ulist1 = trainMatrix.getRows(itemId1);//所有购买item的用户
            if (ulist1.size() < 10) {
                continue;
            }
            for (MatrixEntry me2 : trainMatrix) {
                int itemId2 = me2.column();
                if (itemId2 == itemId1) {
                    continue;
                }
                ulist2 = trainMatrix.getRows(itemId2);
                len = countsame(ulist1, ulist2);
                if (len > 0) {
                    float dnumItems = numItems;
                    float dlen = len;
                    float us1 = ulist1.size();
                    float us2 = ulist2.size();
                    a = ((dnumItems * dlen) / (us1 * us2));
                    double d = Math.log(a) / Math.log(2) - Math.log(5) / Math.log(2);
                    BigDecimal b = new BigDecimal(d);
                    double d1 = b.setScale(2, BigDecimal.ROUND_HALF_UP).doubleValue();
                    if (d1 > 0) {
                        d1 = d1;
                    } else {
                        d1 = 0;
                    }
                    if (d1 > 0) {
                        if (maxd < d1) {
                            maxd = d1;
                        }
                        SPPMImatrix.add1(itemId1, itemId2, d1);
                        SPPMImatrix.add1(itemId2, itemId1, d1);
                    }
                }
            }
        }
        // 归一化
        int woshidj=0;
        for (int i = 0; i < numItems; i++) {
            if(i%150==0)
            {
                System.out.println("归一化  "+(woshidj++));
            }
            for (int j = 0; j < numItems; j++) {
                double p = SPPMImatrix.get(i, j);
                double newp = p / maxd;
                BigDecimal b = new BigDecimal(newp);
                double newp1 = b.setScale(2, BigDecimal.ROUND_HALF_UP).doubleValue();
                SPPMImatrix.add1(i, j, newp1);
            }
        }
        //写 SPPMI矩阵到txt中   A2
        File file = new File("D:/COFACTORRecommender.txt");
        try {
            file.createNewFile();
        } catch (IOException e) {
            e.printStackTrace();
        }
        // 格式化浮点数据
        NumberFormat formatter = NumberFormat.getNumberInstance();
        formatter.setMaximumFractionDigits(5); // 设置最大小数位为5
        PrintWriter pfp = null; //设置输出文件的编码为utf-8
        try {
            pfp = new PrintWriter(file, "UTF-8");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
        StringBuffer thisLine = new StringBuffer("");
        String field = "";
        for (int i1 = 0; i1 < numItems; i1++) {
            for (int j1 = 0; j1 < numItems; j1++) {
                field = formatter.format(SPPMImatrix.get(i1, j1));
                if (j1 < numItems - 1) {
                    thisLine.append(field).append("\t");
                } else {
                    thisLine.append(field);
                }
            }
            pfp.print(thisLine.toString() + "\n");
            thisLine = new StringBuffer("");
        }
        pfp.close();
    }

//        //  读文件   A3
//            StringBuffer sb = new StringBuffer("");
//            File file1 = new File("D:/COFACTORRecommender.txt");
//            try {
//                InputStreamReader read = new InputStreamReader(new FileInputStream(file1), "utf-8");
//                BufferedReader br = new BufferedReader(read);
//                String str = null;
//                int i3 = -1;//行
//                while ((str = br.readLine()) != null) {  //str 一行数据
//
//                    String[] strarr = str.split("\t");
//                    //List<String> result=new ArrayList<String>();
//                    i3++;
//                    for (int i2 = 0; i2 < strarr.length; i2++)  //i2 列
//                    {
//
//                        double hehe = Double.parseDouble(strarr[i2]);
//                        SPPMImatrix.add1(i3, i2, hehe);
//                    }
//                }
//                read.close();
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
//        }

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
                int userId = me.row();
                int itemId = me.column();
                double realRating = me.get();
                double predictRating = predict(userId, itemId);
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

            double diff=0.0d;
            for(int i=0;i<numItems;i++)
            {
                for(int j=0;j<numItems;j++)
                {
                    double mij=SPPMImatrix.get(i,j);
                    if(mij==0)
                    {
                        continue;
                    }

//                    protected double predict1(int i, int j) throws LibrecException {
//                    return DenseMatrix.rowMult( itemFactors, i,g,j); 写在父类里面
//                }
                    double predictrating2=predict1(i,j);
                    double wi=w.get(i);
                    double cj=c.get(j);
                    diff=mij-predictrating2-wi-cj;
                    loss+=diff*diff;
                    w.add(i,learnRate*diff);
                    c.add(j,learnRate*diff);
                    for(int k=0;k<numFactors;k++)
                    {
                        itemFactors.add(i,k,learnRate*diff*g.get(j,k));
                        g.add(j,k,learnRate*diff*itemFactors.get(i,k));
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
