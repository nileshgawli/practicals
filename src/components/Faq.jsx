import React from "react";
import {
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Box,
} from "@chakra-ui/react";

const Faq = () => {
  return (
    <>
      <div className="text-4xl lg:text-5xl font-bold font-Onest text-center my-16">
        DWM
      </div>

      <Box className="font-Onest mx-8 md:mx-20 lg:mx-44 text-content">
        <Accordion allowToggle>
          {/* FAQS */}
          <pre>
            <code>
            <Items
                question={"CRC CN"}
                answer={`
                #include<stdio.h>
                #include<string.h>
                // length of the generator polynomial
                #define N strlen(gen_poly)
                // data to be transmitted and received
                char data[28];
                // CRC value
                char check_value[28];
                // generator polynomial
                char gen_poly[10];
                // variables
                int data_length,i,j;
                // function that performs XOR operation
                void XOR(){
                // if both bits are the same, the output is 0
                // if the bits are different the output is 1
                for(j = 1;j < N; j++)
                check_value[j] = (( check_value[j] == gen_poly[j])?'0':'1');
                }
                // Function to check for errors on the receiver side
                void receiver(){
                // get the received data
                printf("Enter the received data: ");
                scanf("%s", data);
                // Cyclic Redundancy Check
                crc();
                // Check if the remainder is zero to find the error
                for(i=0;(i<N-1) && (check_value[i]!='1');i++);
                if(i<N-1)
                {
                printf("\n\n Error detected\n");
                printf(" ************\n");
                }
                else
                {
                printf("\n\n No error detected\n");
                printf(" ***************\n");
                }
                }
                void crc(){
                // initializing check_value
                for(i=0;i<N;i++)
                check_value[i]=data[i];
                do{
                // check if the first bit is 1 and calls XOR function
                if(check_value[0]=='1')
                XOR();
                // Move the bits by 1 position for the next computation
                for(j=0;j<N-1;j++)
                check_value[j]=check_value[j+1];
                // appending a bit from data
                check_value[j]=data[i++];
                }while(i<=data_length+N-1);
                // loop until the data ends
                }
                int main()
                {
                printf("\nEnter data to be transmitted: ");
                scanf("%s",data);
                printf("\n Enter the Generating polynomial: ");
                // get the generator polynomial
                scanf("%s",gen_poly);
                // find the length of data
                data_length=strlen(data);
                // appending n-1 zeros to the data
                for(i=data_length;i<data_length+N-1;i++)
                data[i]='0';
                printf("\n ");
                // print the data with padded zeros
                printf("\n Data padded with n-1 zeros : %s",data);
                // Cyclic Redundancy Check
                crc();
                // print the computed check value
                printf("\n CRC or Check value is : %s",check_value);
                // Append data with check_value(CRC)
                for(i=data_length;i<data_length+N-1;i++)
                data[i]=check_value[i-data_length];
                // printing the final data to be sent
                printf("\n Final data to be sent : %s",data);
                printf("\n \n");
                // Calling the receiver function to check errors
                receiver();
                return 0;
                }
                `}
              />
            </code>
          </pre>
          <pre>
            <code>
              <Items
                question={"SOCKET CN"}
                answer={`

                SERVER:
                import socket

if __name__ == "__main__":
    ip = "127.0.0.1"
    port = 3344

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((ip,port))
    server.listen(5)

    while True:
        client, address = server.accept()
        print("Client connected from",address)

        string = client.recv(1024)
        string = string.decode("utf-8")
        string = string.upper()
        client.send(bytes(string, "utf-8"))

        client.close()

                CLIENT:
                import socket

if __name__ == "__main__":
    ip = "127.0.0.1"
    port = 3344

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.connect((ip,port))

    string = input("Enter string: ")
    server.send(bytes(string, "utf-8"))
    buffer = server.recv(1024)
    buffer = buffer.decode("utf-8")
    print(f"Server:{buffer} ")
`}
              />
            </code>
          </pre>
          <pre>
            <code>
              <Items
                question={"Page Rank"}
                answer={`
              import java.util.*;
              import java.io.*;
              public class PageRank {

               public int path[][] = new int[10][10];
               public double pagerank[] = new double[10];

               public void calc(double totalNodes) {

                double InitialPageRank;
                double OutgoingLinks = 0;
                double DampingFactor = 0.85;
                double TempPageRank[] = new double[10];
                int ExternalNodeNumber;
                int InternalNodeNumber;
                int k = 1; // For Traversing
                int ITERATION_STEP = 1;
                InitialPageRank = 1 / totalNodes;
                System.out.printf(" Total Number of Nodes :" + totalNodes + "\t Initial PageRank  of All Nodes :" + InitialPageRank + "\n");

                // 0th ITERATION  _ OR _ INITIALIZATION PHASE //

                for (k = 1; k <= totalNodes; k++) {
                 this.pagerank[k] = InitialPageRank;
                }

                System.out.printf("\n Initial PageRank Values , 0th Step \n");
                for (k = 1; k <= totalNodes; k++) {
                 System.out.printf(" Page Rank of " + k + " is :\t" + this.pagerank[k] + "\n");
                }

                while (ITERATION_STEP <= 2) // Iterations
                {
                 // Store the PageRank for All Nodes in Temporary Array
                 for (k = 1; k <= totalNodes; k++) {
                  TempPageRank[k] = this.pagerank[k];
                  this.pagerank[k] = 0;
                 }

                 for (InternalNodeNumber = 1; InternalNodeNumber <= totalNodes; InternalNodeNumber++) {
                  for (ExternalNodeNumber = 1; ExternalNodeNumber <= totalNodes; ExternalNodeNumber++) {
                   if (this.path[ExternalNodeNumber][InternalNodeNumber] == 1) {
                    k = 1;
                    OutgoingLinks = 0; // Count the Number of Outgoing Links for each ExternalNodeNumber
                    while (k <= totalNodes) {
                     if (this.path[ExternalNodeNumber][k] == 1) {
                      OutgoingLinks = OutgoingLinks + 1; // Counter for Outgoing Links
                     }
                     k = k + 1;
                    }
                    // Calculate PageRank
                    this.pagerank[InternalNodeNumber] += TempPageRank[ExternalNodeNumber] * (1 / OutgoingLinks);
                   }
                  }
                 }

                 System.out.printf("\n After " + ITERATION_STEP + "th Step \n");

                 for (k = 1; k <= totalNodes; k++)
                  System.out.printf(" Page Rank of " + k + " is :\t" + this.pagerank[k] + "\n");

                 ITERATION_STEP = ITERATION_STEP + 1;
                }
                // Add the Damping Factor to PageRank
                for (k = 1; k <= totalNodes; k++) {
                 this.pagerank[k] = (1 - DampingFactor) + DampingFactor * this.pagerank[k];
                }

                // Display PageRank
                System.out.printf("\n Final Page Rank : \n");
                for (k = 1; k <= totalNodes; k++) {
                 System.out.printf(" Page Rank of " + k + " is :\t" + this.pagerank[k] + "\n");
                }

               }

               public static void main(String args[]) {
                int nodes, i, j, cost;
                Scanner in = new Scanner(System.in);
                System.out.println("Enter the Number of WebPages \n");
                nodes = in .nextInt();
                PageRank p = new PageRank();
                System.out.println("Enter the Adjacency Matrix with 1->PATH & 0->NO PATH Between two WebPages: \n");
                for (i = 1; i <= nodes; i++)
                 for (j = 1; j <= nodes; j++) {
                  p.path[i][j] = in .nextInt();
                  if (j == i)
                   p.path[i][j] = 0;
                 }
                p.calc(nodes);

               }
              }`}
              />
            </code>
          </pre>
          <pre>
            <code>
              <Items
                question={"Discretization & Visualization"}
                answer={`
              def discretize_and_smooth(data, number_of_bins):

    sorted_data = sorted(data)

    min_value = sorted_data[0]
    max_value = sorted_data[-1]
    bin_width = (max_value - min_value) / number_of_bins
    bin_boundaries = [min_value + i * bin_width for i in range(number_of_bins)]


    bins = {i: [] for i in range(number_of_bins)}
    for value in sorted_data:
        bin_index = int((value - min_value) // bin_width)
        bins[bin_index].append(value)

    bin_means = {i: sum(bin_values) / len(bin_values) for i, bin_values in bins.items()}

    #  Smoothing
    for i, bin_values in bins.items():
        bins[i] = [bin_means[i]] * len(bin_values)
    discretized_data = [value for bin_values in bins.values() for value in bin_values]
    return discretized_data



data = [10, 15, 20, 25, 30, 35, 40, 45, 50]
number_of_bins = 3
discretized_data = discretize_and_smooth(data, number_of_bins)
print(discretized_data)


import matplotlib.pyplot as plt

data = [1, 1, 5, 5, 5, 5, 5, 8, 8, 10, 10, 10, 10, 12, 14, 14, 14, 15, 15, 15, 15, 15, 15, 18, 18, 18, 18, 18, 18, 18, 18, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 22, 23, 23,23,23,23,24,24, 25, 25, 25, 28, 28, 30, 30, 30]

num_bins = 10

# Create the histogram
plt.hist(data, bins=num_bins, color="#C4A484", edgecolor="black", alpha=0.7)  # Using alpha for transparency

# Add labels and title
plt.xlabel('Price')
plt.ylabel('Count')
plt.title('Histogram of Data')

# Display the histogram
plt.show()

# Create the scatterplot
x_values = [i for i in range(len(data))]
plt.scatter(x_values, data, color='green', marker='o', label='Data Points', alpha=0.7)

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Scatterplot of Data')

# Display the scatterplot
plt.legend()
plt.show()
`}
              />
            </code>
          </pre>
          <pre>
            <code>
              <Items
                question={"NAIVE BAYESIAN"}
                answer={`
              import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'Age': ['Young', 'Young', 'Middle-aged', 'Middle-aged', 'Senior', 'Young', 'Middle-aged', 'Young', 'Senior', 'Middle-aged', 'Young', 'Middle-aged', 'Middle-aged', 'Senior'],
    'Income': ['Low', 'High', 'Medium', 'Medium', 'Low', 'High', 'Medium', 'Medium', 'High', 'Low', 'Medium', 'Medium', 'High', 'Low'],
    'Product': ['No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

df_encoded = pd.get_dummies(df.iloc[:, :-1])

X = df_encoded.values
y = df['Product'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=87)

gaussian_naive_bayes = GaussianNB()
gaussian_naive_bayes.fit(X_train, y_train)

y_pred = gaussian_naive_bayes.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

gender_input = input("Enter gender (Male/Female): ")
age_input = input("Enter age (Young/Middle-aged/Senior): ")
income_input = input("Enter income (Low/Medium/High): ")

user_input = pd.DataFrame({
    'Gender_' + gender_input: 1,
    'Age_' + age_input: 1,
    'Income_' + income_input: 1
}, index=[0])

user_input_encoded = user_input.reindex(columns=df_encoded.columns, fill_value=0)
prediction = gaussian_naive_bayes.predict(user_input_encoded.values)
print(prediction)`}
              />
            </code>
          </pre>
          <pre>
            <code>
              <Items
                question={"KMEAN"}
                answer={`
              import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class KMeans {

    public static void main(String[] args) {
        // Number of clusters
        int k = 3;

        // Number of data points
        int n = 100;

        // Generate some random data points
        List<Point> data = generateData(n);

        // Initialize the centroids
        List<Point> centroids = initializeCentroids(data, k);

        // K-Means clustering
        List<List<Point>> clusters = kMeans(data, centroids, k, 100);

        // Print the clusters
        for (int i = 0; i < k; i++) {
            System.out.println("Cluster " + (i + 1));
            for (Point point : clusters.get(i)) {
                System.out.println(point);
            }
            System.out.println();
        }
    }

    public static List<Point> generateData(int n) {
        List<Point> data = new ArrayList<>();
        Random random = new Random();
        for (int i = 0; i < n; i++) {
            double x = random.nextDouble() * 100;
            double y = random.nextDouble() * 100;
            data.add(new Point(x, y));
        }
        return data;
    }

    public static List<Point> initializeCentroids(List<Point> data, int k) {
        List<Point> centroids = new ArrayList<>();
        Random random = new Random();
        for (int i = 0; i < k; i++) {
            int randomIndex = random.nextInt(data.size());
            centroids.add(data.get(randomIndex));
        }
        return centroids;
    }

    public static List<List<Point>> kMeans(List<Point> data, List<Point> centroids, int k, int maxIterations) {
        List<List<Point>> clusters = new ArrayList<>();

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            // Initialize empty clusters
            clusters.clear();
            for (int i = 0; i < k; i++) {
                clusters.add(new ArrayList<>());
            }

            // Assign data points to the nearest centroid
            for (Point point : data) {
                int nearestCentroidIndex = findNearestCentroid(point, centroids);
                clusters.get(nearestCentroidIndex).add(point);
            }

            // Update centroids
            for (int i = 0; i < k; i++) {
                centroids.set(i, calculateCentroid(clusters.get(i)));
            }
        }

        return clusters;
    }

    public static int findNearestCentroid(Point point, List<Point> centroids) {
        int nearestCentroidIndex = 0;
        double minDistance = Double.MAX_VALUE;
        for (int i = 0; i < centroids.size(); i++) {
            double distance = point.distanceTo(centroids.get(i));
            if (distance < minDistance) {
                minDistance = distance;
                nearestCentroidIndex = i;
            }
        }
        return nearestCentroidIndex;
    }

    public static Point calculateCentroid(List<Point> cluster) {
        double sumX = 0.0;
        double sumY = 0.0;
        for (Point point : cluster) {
            sumX += point.getX();
            sumY += point.getY();
        }
        int size = cluster.size();
        return new Point(sumX / size, sumY / size);
    }

    static class Point {
        private double x;
        private double y;

        public Point(double x, double y) {
            this.x = x;
            this.y = y;
        }

        public double getX() {
            return x;
        }

        public double getY() {
            return y;
        }

        public double distanceTo(Point other) {
            double dx = x - other.x;
            double dy = y - other.y;
            return Math.sqrt(dx * dx + dy * dy);
        }

        @Override
        public String toString() {
            return "(" + x + ", " + y + ")";
        }
    }
}
`}
              />
            </code>
          </pre>
          <pre>
            <code>
              <Items
                question={"KMEAN SHORT PROGRAM"}
                answer={`
                public class KMeansClustering {

                  public static void main(String args[]) {
                      int arr[] = {2, 4, 10, 12, 3, 20, 30, 11, 25};    // initial data
                      int i, m1, m2, a, b, n = 0;
                      boolean flag;
                      float sum1, sum2;
                      a = arr[0];
                      b = arr[1];
                      m1 = a;
                      m2 = b;
                      int cluster1[] = new int[arr.length], cluster2[] = new int[arr.length];
                      do {
                          sum1 = 0;
                          sum2 = 0;
                          cluster1 = new int[arr.length];
                          cluster2 = new int[arr.length];
                          n++;
                          int k = 0, j = 0;
                          for (i = 0; i < arr.length; i++) {
                              if (Math.abs(arr[i] - m1) <= Math.abs(arr[i] - m2)) {
                                  cluster1[k] = arr[i];
                                  k++;
                              } else {
                                  cluster2[j] = arr[i];
                                  j++;
                              }
                          }
                          System.out.println();
                          for (i = 0; i < k; i++) {
                              sum1 = sum1 + cluster1[i];
                          }
                          for (i = 0; i < j; i++) {
                              sum2 = sum2 + cluster2[i];
                          }
                          //printing Centroids/Means\
                          System.out.println("m1=" + m1 + "   m2=" + m2);
                          a = m1;
                          b = m2;
                          m1 = Math.round(sum1 / k);
                          m2 = Math.round(sum2 / j);
                          flag = !(m1 == a && m2 == b);

                          System.out.println("After iteration " + n + " , cluster 1 :\n");    //printing the clusters of each iteration
                          for (i = 0; i < cluster1.length; i++) {
                              System.out.print(cluster1[i] + "\t");
                          }

                          System.out.println("\n");
                          System.out.println("After iteration " + n + " , cluster 2 :\n");
                          for (i = 0; i < cluster2.length; i++) {
                              System.out.print(cluster2[i] + "\t");
                          }

                      } while (flag);

                      System.out.println("Final cluster 1 :\n");            // final clusters
                      for (i = 0; i < cluster1.length; i++) {
                          System.out.print(cluster1[i] + "\t");
                      }

                      System.out.println();
                      System.out.println("Final cluster 2 :\n");
                      for (i = 0; i < cluster2.length; i++) {
                          System.out.print(cluster2[i] + "\t");
                      }
                  }

                  }`}
              />
            </code>
          </pre>
          <pre>
            <code>
              <Items
                question={"STAR, SNOWFLAKE SCHEMA"}
                answer={`
              CREATE TABLE DEPARTMENTS (
                dept_id number primary key,
                dept_name varchar(20),
                dept_head varchar(20)
               );
               INSERT INTO DEPARTMENTS VALUES(101, 'Accounts', 'Mr. Sarthak Shah');
               INSERT INTO DEPARTMENTS VALUES(102, 'Loan', 'Ms. Anandi Vartak');
               INSERT INTO DEPARTMENTS VALUES(103, 'Finance', 'Mr. Subodh Patil');
               INSERT INTO DEPARTMENTS VALUES(104, 'Cashier', 'Mr. Soham katdare');
               SELECT * FROM DEPARTMENTS;
               CREATE TABLE EMPLOYEES (
                emp_id number primary key,
                emp_name varchar(20),
                emp_designation varchar(20),
                emp_salary number
               );
               INSERT INTO EMPLOYEES VALUES(11, 'Ms. Ruhi', 'Loan officer', 40000);
               INSERT INTO EMPLOYEES VALUES(12, 'Mr. Samarth', 'Bank Manager', 100000);
               INSERT INTO EMPLOYEES VALUES(13, 'Mr. Swapnil', 'Financial accountant', 40000);
               INSERT INTO EMPLOYEES VALUES(14, 'Ms. Shriya', 'Bank teller', 50000);
               SELECT * FROM EMPLOYEES;
               CREATE TABLE CUSTOMER (
                cust_id number primary key,
                cust_name varchar(20),
                cust_querry varchar(100),
                cust_querry_status varchar(20)
               );
               INSERT INTO CUSTOMER VALUES(1, 'Mr. Swapnil Wade', 'Passbook', 'Solved');
               INSERT INTO CUSTOMER VALUES(2, 'Ms. Gaytri', 'ATM Card', '');
               INSERT INTO CUSTOMER VALUES(3, 'Mr. Yash korla', 'Loan', 'Forwarded to dept');
               INSERT INTO CUSTOMER VALUES(4, 'Ms. Salma Shaikh', 'KYC', 'Don');
               SELECT * FROM CUSTOMER;
               CREATE TABLE LOCATIONS (
                location_id number primary key,
                location_name varchar(20),
                city varchar(20),
                state varchar(20)
               );
               INSERT INTO LOCATIONS VALUES(111, 'Kisan nagar', 'Thane', 'Maharashtra');
               INSERT INTO LOCATIONS VALUES(112, 'Vartak nagar', 'Thane', 'Maharashtra');
               INSERT INTO LOCATIONS VALUES(113, 'Mahada Colony', 'Mumbai', 'Maharashtra');
               INSERT INTO LOCATIONS VALUES(114, 'Vasant Vinhar', 'Mumbai', 'Maharashtra');
               SELECT * FROM LOCATIONS;
               CREATE TABLE BRANCH (
                branch_id number primary key,
                branch_name varchar(20),
                branch_manager varchar(20)
               );
               INSERT INTO BRANCH VALUES(111, 'Kisan nagar', 'Mr. Soham Dixit');
               INSERT INTO BRANCH VALUES(112, 'Vartak nagar', 'Mr. Aryan Surve');
               INSERT INTO BRANCH VALUES(113, 'Mahada Colony', 'Mr. Dhruv Sheth');
               INSERT INTO BRANCH VALUES(114, 'Vasant Vinhar', 'Mr. manish Patil');
               SELECT * FROM BRANCH;
               CREATE TABLE BANK (
                bank_id number primary key,
                bank_name varchar(20),
                location_id number,
                dept_id number,
                emp_id number,
                cust_id number,
                branch_id number,
                FOREIGN KEY (location_id) REFERENCES LOCATIONS(location_id),
                FOREIGN KEY (dept_id) REFERENCES DEPARTMENTS(dept_id),
                FOREIGN KEY (emp_id) REFERENCES EMPLOYEES(emp_id),
                FOREIGN KEY (cust_id) REFERENCES CUSTOMER(cust_id),
                FOREIGN KEY (branch_id) REFERENCES BRANCH(branch_id)
               );`}
              />
            </code>
          </pre>
          <pre>
            <code>
              <Items
                question={"OLAP Operations"}
                answer={`
              ROLLUP-
              SELECT account_type, SUM (balance) AS TOTAL_BALANCE FROM ACCOUNTS GROUP BY ROLLUP (account_type)

              DRILL DOWN-
              SELECT account_type, account_id, SUM(balance) AS TOTAL_BALANCE
              FROM ACCOUNTS
              GROUP BY GROUPING SETS((account_type, account_id), (account_type))

              SLICE-
              SELECT * FROM ACCOUNTS WHERE customer_id IN (SELECT customer_id FROM CUSTOMERS WHERE age = 35)

              DICE-
              SELECT * FROM ACCOUNTS WHERE account_type = "Loan" AND customer_id = 11
              `}
              />
            </code>
          </pre>
          <pre>
            <code>
              <Items
                question={"OLAP Script"}
                answer={`
                create table CLASS_INFO(
                  class_id int primary key,
                  class_name varchar(30),
                 address varchar(50)
                 );
                 create table CR(
                  cr_id int primary key,
                  cr_name varchar(30),
                 cr_contno int,
                  cr_mail varchar(30)
                 );
                 create table HOD(
                  hod_id int primary key,
                  hod_name varchar(30),
                 hod_address varchar(50),
                 hod_contno int,
                 hod_mail varchar(30)
                 );
                 create table TEACHER(
                  teacher_id int primary key,
                  teacher_name varchar(30),
                 teacher_contno int,
                 teacher_address varchar(50),
                 teacher_exp varchar(30)
                 );
                 create table STUDENT(
                  student_id int primary key,
                  student_name varchar(30),
                 student_address varchar(50),
                 student_contno int
                 );
                 create table CLASS_TABLE(
                 class_id int,
                  hod_id int,
                  teacher_id int,
                  foreign key(class_id) references CLASS_INFO(class_id),
                  foreign key(hod_id) references HOD(hod_id),
                  foreign key(teacher_id) references TEACHER(teacher_id)
                 );
                 insert into CLASS_INFO values('458', 'COMPS', 'Mumbai');
                 insert into CLASS_INFO values('759', 'EXTC', 'Nerul');
                 insert into CLASS_INFO values('366', 'AIDS', 'Seawoods');
                 insert into CLASS_INFO values('476', 'MECH', 'Kharkopar');
                 insert into CR values('126', 'Rohit', '8766543876', 'rohit12@gmail.com');
                 insert into CR values ('434', 'Ashish', '8766556876', 'ash44@gmail.com');
                 insert into CR values('553', 'Atharv', '8766654876', 'ath66@gmail.com');
                 insert into CR values('67', 'Rakesh', '8766556477', 'rakesh20@gmail.com');
                 insert into HOD values('332', 'Varun', 'Manor', '9865453542','var33@gmail.com');
                 insert into HOD values('352', 'Harsh', 'Wada', '7865453542','harsh56@gmail.com');
                 insert into HOD values('42', 'Avinash', 'Palghar', '8865453542','avi45@gmail.com');
                 insert into HOD values('98', 'Nikhil', 'Panvel', '7855453542','nikhil76@gmail.com');
                 insert into TEACHER values('23', 'Mathan', '7654874432', 'Kharghar', 'Nutritionist');
                 insert into TEACHER values('432', 'Neel', '9854874432', 'Kalwa', 'cardio');
                 insert into TEACHER values('12', 'pranit', '7784874432', 'Thane', 'calistehnics');
                 insert into TEACHER values('22', 'Saurabh', '7654324432', 'Airoli', 'cardio');
                 insert into STUDENT values('534', 'Mayur', 'Nerul', '7655439821');
                 insert into STUDENT values('1', 'Jay', 'Vashi', '7656439821');
                 insert into STUDENT values('3', 'Aakash', 'Sanapada', '9345439821');
                 insert into STUDENT values('9', 'Pritesh', 'Virar', '8855439821');
                 insert into CLASS_TABLE values('458', '332', '23');
                 insert into CLASS_TABLE values('759', '352', '23');
                 insert into CLASS_TABLE values('366', '42', '23');
                 insert into CLASS_TABLE values('476', '98', '23');
                 select * from CLASS_INFO;
                 select * from CR;
                 select * from HOD;
                 select * from TEACHER;
                 select * from STUDENT;
                 select * from CLASS_TABLE;


                 1 . Slice Operation:
                 select class_id
                 from CLASS_TABLE
                 where HOD_id ='332'

                 2. Dice Operation:
                 select class_id
                 from CLASS_TABLE
                 where hod_id ='332'
                 and teacher_id = '23'

                 3. Roll- Up Operation:
                 select class_id, sum(hod_id) as hod_id
                 from CLASS_TABLE
                 group by class_id

                 4. Drill- Down Operation:
                 select class_id, teacher_id from CLASS_TABLE

                 5. Drill- Across Operation:
                 select c.class_id, c.hod_id, c.teacher_id , h.hod_name
                 from CLASS_TABLE c
                 join HOD h on c.hod_id = h.hod_id
                 where c.hod_id ='42'

                 6.PIVOT:
                 select *
                 from (
                 select h.hod_id, c.class_id
                  from CLASS_TABLE c
                  inner join HOD h on c.hod_id = h.hod_id

                 )
                 PIVOT(
                  sum(class_id)
                  for hod_id in ('332' as "Tanjiro", '352' as "Nezuko", '42' as "Inosuke", '98' as "Zenistu" )
                 )
              `}
              />
            </code>
          </pre>
        </Accordion>
      </Box>
    </>
  );
};

const Items = (props) => {
  return (
    <>
      {/* <AccordionItem className="bg-zinc-200 py-2 px-4 rounded border-2 border-b-zinc-400"> */}
      <AccordionItem className="my-2 bg-zinc-100 py-2 lg:py-4 px-4 lg:px-8 rounded border-2 border-zinc-400">
        <h2>
          <AccordionButton>
            <Box
              className="text-lg lg:text-xl"
              as="span"
              flex="1"
              textAlign="left"
            >
              {props.question}
            </Box>
            <Box className="text-3xl">
              <AccordionIcon />
            </Box>
          </AccordionButton>
        </h2>
        <AccordionPanel
          className="text-base lg:text-lg text-zinc-600 mt-2"
          pb={4}
        >
          {props.answer}
        </AccordionPanel>
      </AccordionItem>
    </>
  );
};

export default Faq;
