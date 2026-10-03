import java.io.*;
import java.nio.charset.StandardCharsets;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Reads data/twcs.csv with a real CSV parser (quotes, commas and multi-line
 * tweets are handled), extracts hashtags + tweet date and writes one
 * "date,hashtag" line per hashtag occurrence to output/hashtags.csv.
 *
 * Run from the project root:
 *   java TwitterDataIngestion [inputCsv] [outputCsv]
 */
public class TwitterDataIngestion {

    private static final Pattern HASHTAG = Pattern.compile("#([\\p{L}\\p{N}_]+)");
    // Example value: Tue Oct 31 22:10:47 +0000 2017
    private static final DateTimeFormatter TWITTER_DATE =
            DateTimeFormatter.ofPattern("EEE MMM dd HH:mm:ss Z yyyy", Locale.ENGLISH);

    /** Reads one CSV record (may span several physical lines). Returns null at end of file. */
    static List<String> readRecord(BufferedReader r) throws IOException {
        List<String> record = new ArrayList<>();
        StringBuilder field = new StringBuilder();
        boolean inQuotes = false;
        boolean any = false;
        int c;
        while ((c = r.read()) != -1) {
            any = true;
            char ch = (char) c;
            if (inQuotes) {
                if (ch == '"') {
                    r.mark(1);
                    int next = r.read();
                    if (next == '"') {
                        field.append('"');          // escaped quote ""
                    } else {
                        inQuotes = false;           // closing quote
                        if (next != -1) r.reset();
                    }
                } else {
                    field.append(ch);               // commas / newlines inside quotes are text
                }
            } else {
                if (ch == '"') {
                    inQuotes = true;
                } else if (ch == ',') {
                    record.add(field.toString());
                    field.setLength(0);
                } else if (ch == '\n') {
                    record.add(field.toString());
                    return record;
                } else if (ch != '\r') {
                    field.append(ch);
                }
            }
        }
        if (!any) return null;
        record.add(field.toString());
        return record;
    }

    public static void main(String[] args) {
        String inputFile = args.length > 0 ? args[0] : "data/twcs.csv";
        String outputFile = args.length > 1 ? args[1] : "output/hashtags.csv";

        File in = new File(inputFile);
        if (!in.exists()) {
            System.err.println("ERROR: dataset not found: " + in.getAbsolutePath());
            System.err.println("Run this program from the project root (folder that contains data/twcs.csv).");
            System.exit(1);
        }
        File outDir = new File(outputFile).getAbsoluteFile().getParentFile();
        if (outDir != null && !outDir.exists()) outDir.mkdirs();

        long records = 0, invalid = 0, badDate = 0, tweetsWithTags = 0, tags = 0;

        try (BufferedReader reader = new BufferedReader(
                     new InputStreamReader(new FileInputStream(in), StandardCharsets.UTF_8), 1 << 16);
             BufferedWriter writer = new BufferedWriter(
                     new OutputStreamWriter(new FileOutputStream(outputFile), StandardCharsets.UTF_8), 1 << 16)) {

            List<String> header = readRecord(reader);
            if (header == null) {
                System.err.println("ERROR: CSV file is empty.");
                System.exit(1);
            }
            int dateIdx = header.indexOf("created_at");
            int textIdx = header.indexOf("text");
            if (dateIdx < 0 || textIdx < 0) {
                System.err.println("ERROR: CSV must contain 'created_at' and 'text' columns. Found: " + header);
                System.exit(1);
            }
            int needed = Math.max(dateIdx, textIdx) + 1;

            List<String> rec;
            while ((rec = readRecord(reader)) != null) {
                if (rec.size() < needed || (rec.size() == 1 && rec.get(0).isEmpty())) {
                    invalid++;
                    continue;
                }
                records++;

                Matcher m = HASHTAG.matcher(rec.get(textIdx));
                if (m.find()) {
                    String date;
                    try {
                        date = ZonedDateTime.parse(rec.get(dateIdx).trim(), TWITTER_DATE)
                                .withZoneSameInstant(ZoneOffset.UTC).toLocalDate().toString();
                    } catch (Exception e) {
                        badDate++;
                        continue;
                    }
                    do {
                        String tag = m.group(1).toLowerCase(Locale.ROOT);
                        writer.write(date);
                        writer.write(',');
                        writer.write(tag);
                        writer.newLine();
                        tags++;
                    } while (m.find());
                    tweetsWithTags++;
                }

                if (records % 100000 == 0) {
                    System.out.println("Processed tweets: " + records + " | hashtags found: " + tags);
                }
            }
        } catch (IOException e) {
            System.err.println("ERROR while processing dataset: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }

        System.out.println("\n========== INGESTION COMPLETE ==========");
        System.out.println("Tweet records parsed        : " + records);
        System.out.println("Invalid/short records       : " + invalid);
        System.out.println("Tweets with hashtags        : " + tweetsWithTags);
        System.out.println("Tweets skipped (bad date)   : " + badDate);
        System.out.println("Hashtag occurrences written : " + tags);
        System.out.println("Output file                 : " + outputFile + "  (format: date,hashtag)");
    }
}
