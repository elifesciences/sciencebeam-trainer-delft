elifePipeline {
    node('containers-jenkins-plugin') {
        def commit
        def baseGrobidTag

        stage 'Checkout', {
            checkout scm
            commit = elifeGitRevision()
            baseGrobidTag = sh(
                script: 'bash -c "source .env && echo \\$BASE_GROBID_TAG"',
                returnStdout: true
            ).trim()
            echo "baseGrobidTag: ${baseGrobidTag}"
            assert baseGrobidTag != ''
            if (env.TAG_NAME) {
                version = env.TAG_NAME - 'v'
            } else {
                version = 'develop'
            }
        }

        stage 'Build and run tests', {
            try {
                sh "make IMAGE_TAG=${commit} REVISION=${commit} ci-build-and-test"
            } finally {
                sh "make ci-clean"
            }
        }

        elifeMainlineOnly {
            stage 'Merge to master', {
                elifeGitMoveToBranch commit, 'master'
            }

            stage 'Push unstable sciencebeam-trainer-delft image', {
                def image = DockerImage.elifesciences(this, 'sciencebeam-trainer-delft', commit)
                def unstable_image = image.addSuffixAndTag('_unstable', commit)
                unstable_image.tag('latest').push()
                unstable_image.push()
            }

            stage 'Push unstable sciencebeam-trainer-delft-grobid image', {
                def tag = "${baseGrobidTag}-${commit}"
                def image = DockerImage.elifesciences(this, 'sciencebeam-trainer-delft-grobid', tag)
                def unstable_image = image.addSuffixAndTag('_unstable', tag)
                unstable_image.tag('latest').push()
                unstable_image.push()
            }

            stage 'Push unstable sciencebeam-trainer-delft-trainer-grobid image', {
                def tag = "${baseGrobidTag}-${commit}"
                def image = DockerImage.elifesciences(this, 'sciencebeam-trainer-delft-trainer-grobid', tag)
                def unstable_image = image.addSuffixAndTag('_unstable', tag)
                unstable_image.tag('latest').push()
                unstable_image.push()
            }
        }

        elifeTagOnly { repoTag ->
            stage 'Push stable sciencebeam-trainer-delft image', {
                def image = DockerImage.elifesciences(this, 'sciencebeam-trainer-delft', commit)
                image.tag('latest').push()
                image.tag(version).push()
            }

            stage 'Push stable sciencebeam-trainer-delft-grobid image', {
                def tag = "${baseGrobidTag}-${commit}"
                def image = DockerImage.elifesciences(this, 'sciencebeam-trainer-delft-grobid', tag)
                image.tag('latest').push()
                image.tag(version).push()
            }

            stage 'Push stable sciencebeam-trainer-delft-trainer-grobid image', {
                def tag = "${baseGrobidTag}-${commit}"
                def image = DockerImage.elifesciences(this, 'sciencebeam-trainer-delft-trainer-grobid', tag)
                image.tag('latest').push()
                image.tag(version).push()
            }
        }
    }
}
