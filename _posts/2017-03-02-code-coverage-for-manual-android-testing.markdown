---
layout: post
title: "How to get Code Coverage during Manual Testing for Android App"
date: 2017-03-02 20:55:00 +0300
categories: ["dev"]
excerpt: "Are you stuck trying to do some manual testing and get the code coverage from it? Here is a simple (but somewhat barebone) solution to that problem."
---

This one should be a quick one, hopefully! Recently, I've been trying to gather coverage data of an app during manual testing. Imagine exercising the app using MonkeyRunner, Ui Automator or some smart AI agent (more about that in later posts). 

Many tools (and papers) describe that these unicorns exist and are easy to use. Yet, my experience suggests that this is not the case. The few I've tried are: [BBoxTester](https://github.com/zyrikby/BBoxTester) (never got it to instrument a custom app), [SwiftHand](https://github.com/wtchoi/SwiftHand/) (same as BBoxTester) and various hacks using [Emma](http://emma.sourceforge.net/). None of it works (but that might be just me).

# How to do it?

Meet [JaCoCo](http://www.jacoco.org/jacoco/), the tool that really works (yes, there is a catch!). Normally, JaCoCo is used for code coverage when you are executing tests. However, a few tweaks will do the trick.

To "install" JaCoCo in your Android project open the *build.gradle* file within the *app* folder and add the following at the top level:

```groovy
def coverageSourceDirs = [
        '../app/src/main/java'
]

jacoco{
    toolVersion = "0.7.6.201602180812" // try a newer version if you can
}
```

Next, let's define a task (in the same file) which will generate HTML report for the code coverage achieved during the testing:

```groovy
task jacocoTestReport(type: JacocoReport) {
    group = "Reporting"
    description = "Generate Jacoco coverage reports after running tests."
    reports {
        xml.enabled = true
        html.enabled = true
    }
    classDirectories = fileTree(
            dir: './build/intermediates/classes/debug',
            excludes: ['**/R*.class',
                       '**/*$InjectAdapter.class',
                       '**/*$ModuleAdapter.class',
                       '**/*$ViewInjector*.class'
            ])
    sourceDirectories = files(coverageSourceDirs)
    executionData = files("$buildDir/outputs/code-coverage/connected/coverage.exec")
    doFirst {
        new File("$buildDir/intermediates/classes/").eachFileRecurse { file ->
            if (file.name.contains('$$')) {
                file.renameTo(file.path.replace('$$', '$'))
            }
        }
    }
}
```

Great! JaCoCo is installed and should be working nicely! Add the following within android -> buildTypes (in the same file):

```groovy
debug {
    testCoverageEnabled = true
}
```

Next, add *resources* directory to app -> src -> main. Add jacoco-agent.properties file to that folder. The file should contain:

```
destfile=/storage/sdcard/coverage.exec
```

The coverage data will be recorded at the device. So, we need a permission to write there. Add the following to your *AndroidManifest.xml* file:

```xml
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
```

On Android 6+ you should request this permission during runtime. Here is a sample:

```java
public static void verifyStoragePermissions(Activity activity) {
        // Check if we have read or write permission
        int writePermission = ActivityCompat.checkSelfPermission(activity, 
                Manifest.permission.WRITE_EXTERNAL_STORAGE);
        int readPermission = ActivityCompat.checkSelfPermission(activity, 
                Manifest.permission.READ_EXTERNAL_STORAGE);

        if (writePermission != PackageManager.PERMISSION_GRANTED || 
            readPermission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission so prompt the user
            ActivityCompat.requestPermissions(
                    activity,
                    PERMISSIONS_STORAGE,
                    REQUEST_EXTERNAL_STORAGE
            );
        }
}
```

Make sure you call this method and obtain the permission before starting/stopping any tests. Next, let's define a helper class which will generate the report file:

```java
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.lang.reflect.Method;

public class JacocoReportGenerator {
    static void generateCoverageReport() {
        String TAG = "jacoco";
        // use reflection to call emma dump coverage method, to avoid
        // always statically compiling against emma jar
        Log.d("StorageSt", Environment.getExternalStorageState());
        String coverageFilePath = Environment.getExternalStorageDirectory() + File.separator + "coverage.exec";
        File coverageFile = new File(coverageFilePath);
        try {
            coverageFile.createNewFile();
            Class<?> emmaRTClass = Class.forName("com.vladium.emma.rt.RT");
            Method dumpCoverageMethod = emmaRTClass.getMethod("dumpCoverageData",
                    coverageFile.getClass(), boolean.class, boolean.class);

            dumpCoverageMethod.invoke(null, coverageFile, false, false);
            Log.e(TAG, "generateCoverageReport: ok");
        } catch (Exception  e) {
            throw new RuntimeException("Is emma jar on classpath?", e)
        }
    }
}
```

Now, it is up to decide where the call to *generateCoverageReport()* should happen. To test it out, put it in some *onPause()* method of an Activity.

Run the app and do your testing. When you are done execute the following *adb* command:

```bash
adb pull /sdcard/coverage.exec app/build/outputs/code-coverage/connected/coverage.exec
```

Make sure you execute it in the root folder of your project and all folders are already created. Finally, generate the report using the task we created:

```bash
./gradlew jacocoTestReport
```

Open the *index.html* file in *app/build/reports/jacoco/jacocoTestReport/html/* folder. Now go grab a cookie and enjoy the victory!

# Is this the best solution?

No! But it is a start. There is no sure way to receive an event when the app is closing/finishing and save the report then. However, some magic tools like [ProbeDroid](https://github.com/ZSShen/ProbeDroid) might offer ways to alleviate that pain. Please, write in the comments below if other, easier, solutions exist!
