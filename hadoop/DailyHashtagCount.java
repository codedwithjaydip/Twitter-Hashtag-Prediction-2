import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/** Job 2: daily frequency.  Input line: date,hashtag   Output: date,hashtag<TAB>count */
public class DailyHashtagCount {

    public static class DailyMapper extends Mapper<Object, Text, Text, IntWritable> {
        private static final IntWritable ONE = new IntWritable(1);
        private final Text outKey = new Text();

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            int comma = line.indexOf(',');
            if (comma <= 0) return;                      // malformed line / missing date
            String date = line.substring(0, comma).trim();
            String tag = line.substring(comma + 1).trim();
            if (date.isEmpty() || tag.isEmpty()) return;
            outKey.set(date + "," + tag);
            context.write(outKey, ONE);
        }
    }

    public static class DailyReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private final IntWritable result = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable v : values) sum += v.get();
            result.set(sum);
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: DailyHashtagCount <input> <output>");
            System.exit(2);
        }
        Job job = Job.getInstance(new Configuration(), "Daily Hashtag Count");
        job.setJarByClass(DailyHashtagCount.class);
        job.setMapperClass(DailyMapper.class);
        job.setCombinerClass(DailyReducer.class);
        job.setReducerClass(DailyReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
